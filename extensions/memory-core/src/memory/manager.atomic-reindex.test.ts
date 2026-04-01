import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { OpenClawConfig } from "openclaw/plugin-sdk/memory-core-host-engine-foundation";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type { MemoryIndexManager } from "./index.js";

let shouldFail = false;

type EmbeddingTestMocksModule = typeof import("./embedding.test-mocks.js");
type TestManagerHelpersModule = typeof import("./test-manager-helpers.js");
type MemoryIndexModule = typeof import("./index.js");
type TestMemorySource = "memory" | "sessions";

describe("memory manager atomic reindex", () => {
  let fixtureRoot = "";
  let caseId = 0;
  let workspaceDir: string;
  let indexPath: string;
  let manager: MemoryIndexManager | null = null;
  let embedBatch: ReturnType<EmbeddingTestMocksModule["getEmbedBatchMock"]>;
  let resetEmbeddingMocks: EmbeddingTestMocksModule["resetEmbeddingMocks"];
  let getRequiredMemoryIndexManager: TestManagerHelpersModule["getRequiredMemoryIndexManager"];
  let closeAllMemorySearchManagers: MemoryIndexModule["closeAllMemorySearchManagers"];

  beforeAll(async () => {
    fixtureRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-mem-atomic-"));
  });

  beforeEach(async () => {
    vi.resetModules();
    const embeddingMocks = await import("./embedding.test-mocks.js");
    embedBatch = embeddingMocks.getEmbedBatchMock();
    resetEmbeddingMocks = embeddingMocks.resetEmbeddingMocks;
    ({ getRequiredMemoryIndexManager } = await import("./test-manager-helpers.js"));
    ({ closeAllMemorySearchManagers } = await import("./index.js"));
    vi.stubEnv("OPENCLAW_TEST_MEMORY_UNSAFE_REINDEX", "0");
    resetEmbeddingMocks();
    shouldFail = false;
    embedBatch.mockImplementation(async (texts: string[]) => {
      if (shouldFail) {
        throw new Error("embedding failure");
      }
      return texts.map((_, index) => [index + 1, 0, 0]);
    });
    workspaceDir = path.join(fixtureRoot, `case-${caseId++}`);
    await fs.mkdir(workspaceDir, { recursive: true });
    indexPath = path.join(workspaceDir, "index.sqlite");
    await fs.mkdir(path.join(workspaceDir, "memory"));
    await fs.writeFile(path.join(workspaceDir, "MEMORY.md"), "Hello memory.");
  });

  afterEach(async () => {
    if (manager) {
      await manager.close();
      manager = null;
    }
    await closeAllMemorySearchManagers();
    vi.unstubAllEnvs();
  });

  afterAll(async () => {
    if (!fixtureRoot) {
      return;
    }
    await fs.rm(fixtureRoot, { recursive: true, force: true });
  });

  const createCfg = (params?: { sources?: TestMemorySource[] }) =>
    ({
      agents: {
        defaults: {
          workspace: workspaceDir,
          memorySearch: {
            provider: "openai",
            model: "mock-embed",
            store: { path: indexPath },
            cache: { enabled: false },
            // Perf: keep test indexes to a single chunk to reduce sqlite work.
            chunking: { tokens: 4000, overlap: 0 },
            sync: { watch: false, onSessionStart: false, onSearch: false },
            ...(params?.sources
              ? {
                  sources: params.sources,
                  experimental: { sessionMemory: true },
                }
              : {}),
          },
        },
        list: [{ id: "main", default: true }],
      },
    }) as OpenClawConfig;

  const createManager = async (params?: { sources?: TestMemorySource[] }) => {
    manager = await getRequiredMemoryIndexManager({
      cfg: createCfg(params),
      agentId: "main",
    });
    return manager;
  };

  const writeTranscript = async (filePath: string, text: string) => {
    await fs.writeFile(
      filePath,
      `${JSON.stringify({
        type: "message",
        message: { role: "user", content: [{ type: "text", text }] },
      })}\n`,
    );
  };

  it("keeps the prior index when a full reindex fails", async () => {
    manager = await createManager();

    await manager.sync({ force: true });
    const beforeStatus = manager.status();
    expect(beforeStatus.chunks).toBeGreaterThan(0);

    shouldFail = true;
    await expect(manager.sync({ force: true })).rejects.toThrow("embedding failure");

    const afterStatus = manager.status();
    expect(afterStatus.chunks).toBeGreaterThan(0);
  });

  it("retries rolled-back memory work after a session-triggered full reindex failure", async () => {
    const stateDir = path.join(fixtureRoot, `state-rollback-memory-${randomUUID()}`);
    const sessionDir = path.join(stateDir, "agents", "main", "sessions");
    const sessionPath = path.join(sessionDir, "rollback-memory.jsonl");
    const previousStateDir = process.env.OPENCLAW_STATE_DIR;
    process.env.OPENCLAW_STATE_DIR = stateDir;

    try {
      await fs.mkdir(sessionDir, { recursive: true });
      await writeTranscript(sessionPath, "rollback memory session v1");

      manager = await createManager({ sources: ["memory", "sessions"] });

      await manager.sync({ force: true });

      const internal = manager as unknown as {
        db: {
          prepare: (sql: string) => {
            get: (path: string, source: string) => { hash: string } | undefined;
          };
        };
        indexFile: (
          entry: { path: string },
          options: { source: "memory" | "sessions" },
        ) => Promise<void>;
      };
      const getMemoryHash = (memoryRelPath: string) =>
        internal.db
          .prepare(`SELECT hash FROM files WHERE path = ? AND source = ?`)
          .get(memoryRelPath, "memory")?.hash;
      const brokenMemoryRelPath = "memory/broken.md";
      const originalBrokenHash = getMemoryHash(brokenMemoryRelPath);

      await fs.writeFile(
        path.join(workspaceDir, "memory", "broken.md"),
        "Broken memory updated before rolled-back reindex.",
      );

      const originalIndexFile = internal.indexFile.bind(manager);
      internal.indexFile = async (entry, options) => {
        if (options.source === "sessions") {
          throw new Error("session reindex failure");
        }
        await originalIndexFile(entry, options);
      };

      await expect(manager.sync({ force: true })).rejects.toThrow("session reindex failure");

      expect(getMemoryHash(brokenMemoryRelPath)).toBe(originalBrokenHash);
      expect(manager.status().dirty).toBe(true);

      internal.indexFile = originalIndexFile;
      await manager.sync({ reason: "retry" });

      expect(getMemoryHash(brokenMemoryRelPath)).not.toBe(originalBrokenHash);
      expect(manager.status().dirty).toBe(false);
    } finally {
      if (previousStateDir === undefined) {
        delete process.env.OPENCLAW_STATE_DIR;
      } else {
        process.env.OPENCLAW_STATE_DIR = previousStateDir;
      }
      await fs.rm(stateDir, { recursive: true, force: true });
    }
  });

  it("forces a full session retry after a rolled-back full reindex stops mid-stream", async () => {
    const stateDir = path.join(fixtureRoot, `state-rollback-sessions-${randomUUID()}`);
    const sessionDir = path.join(stateDir, "agents", "main", "sessions");
    const firstSessionPath = path.join(sessionDir, "001-first.jsonl");
    const secondSessionPath = path.join(sessionDir, "002-second.jsonl");
    const thirdSessionPath = path.join(sessionDir, "003-third.jsonl");
    const previousStateDir = process.env.OPENCLAW_STATE_DIR;
    process.env.OPENCLAW_STATE_DIR = stateDir;

    try {
      await fs.mkdir(sessionDir, { recursive: true });
      await writeTranscript(firstSessionPath, "rollback first transcript v1");
      await writeTranscript(secondSessionPath, "rollback second transcript v1");
      await writeTranscript(thirdSessionPath, "rollback third transcript v1");

      manager = await createManager({ sources: ["sessions"] });

      await manager.sync({ force: true });

      const internal = manager as unknown as {
        sessionsDirty: boolean;
        sessionsDirtyFiles: Set<string>;
        db: {
          prepare: (sql: string) => {
            get: (path: string, source: string) => { hash: string } | undefined;
          };
        };
        indexFile: (
          entry: { path: string },
          options: { source: "memory" | "sessions" },
        ) => Promise<void>;
        getIndexConcurrency: () => number;
      };
      const getSessionHash = (sessionRelPath: string) =>
        internal.db
          .prepare(`SELECT hash FROM files WHERE path = ? AND source = ?`)
          .get(sessionRelPath, "sessions")?.hash;
      const sessionRelPaths = [
        "sessions/001-first.jsonl",
        "sessions/002-second.jsonl",
        "sessions/003-third.jsonl",
      ];
      const originalHashes = new Map(
        sessionRelPaths.map((sessionRelPath) => [sessionRelPath, getSessionHash(sessionRelPath)]),
      );

      await writeTranscript(firstSessionPath, "rollback first transcript v2");
      await writeTranscript(secondSessionPath, "rollback second transcript v2");
      await writeTranscript(thirdSessionPath, "rollback third transcript v2");

      const originalIndexFile = internal.indexFile.bind(manager);
      const originalGetIndexConcurrency = internal.getIndexConcurrency.bind(manager);
      let sessionAttempts = 0;
      internal.getIndexConcurrency = () => 1;
      internal.indexFile = async (entry, options) => {
        if (options.source === "sessions") {
          sessionAttempts += 1;
          if (sessionAttempts === 1) {
            throw new Error("session reindex failure");
          }
        }
        await originalIndexFile(entry, options);
      };

      await expect(manager.sync({ force: true })).rejects.toThrow("session reindex failure");

      for (const sessionRelPath of sessionRelPaths) {
        expect(getSessionHash(sessionRelPath)).toBe(originalHashes.get(sessionRelPath));
      }
      expect(internal.sessionsDirty).toBe(true);
      expect(internal.sessionsDirtyFiles.size).toBe(0);

      internal.indexFile = originalIndexFile;
      internal.getIndexConcurrency = originalGetIndexConcurrency;
      await manager.sync({ reason: "retry" });

      for (const sessionRelPath of sessionRelPaths) {
        expect(getSessionHash(sessionRelPath)).not.toBe(originalHashes.get(sessionRelPath));
      }
      expect(internal.sessionsDirty).toBe(false);
      expect(internal.sessionsDirtyFiles.size).toBe(0);
    } finally {
      if (previousStateDir === undefined) {
        delete process.env.OPENCLAW_STATE_DIR;
      } else {
        process.env.OPENCLAW_STATE_DIR = previousStateDir;
      }
      await fs.rm(stateDir, { recursive: true, force: true });
    }
  });
});
