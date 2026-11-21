import crypto from 'node:crypto';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { Readable } from 'node:stream';
import { describe, test, expect, beforeAll, afterAll, beforeEach, vi } from 'vitest';
import TestStartup from '../utils/test.startup.js';
import { getNotificationService } from '../../server/middleware/app.middleware.js';
import { ROLES, RIGHTS, hasRight } from '../../server/config/rights.js';
import {
    setOllamaClientFactory, resetOllamaClientFactory, createOllamaClient, setLibraryFetch, parseLibrary, parseParameters, parseLibrarySizes, friendlyName,
    mapOllamaError, OllamaError, ERROR_CODES,
    parseNvidiaSmi, appleGpus, parseDisplayAdapters, detectHardware,
    fitModel, fitPool, poolShareMb, splitBySpeed, rankCatalogue, memoryNeededMb, VERDICTS,
    GenerationQueue, extractText, estimateTokens, setFilesFetch, setFilesWakeMs, setSandboxClient, releaseSandboxes, MAX_ATTACHMENT_BYTES,
    isPublicAddress, setRemoteTransport, MAX_REMOTE_BYTES, setSearchFetch,
    poolGroups, subnetOf, privateKind, setToolConnector, ToolSignInError, TOOL_HEALTH, TOOL_CATALOGUE, TOOL_LOCAL_CATALOGUE, CLOUD_PROVIDERS, PROVIDER_HEALTH, setCloudFetch, ollamaOnlineNames, runsOnline, seal, unseal, deviceGateway, DeviceGateway, cloudTitle
} from '../../server/middleware/ai.middleware.js';
import { Device, Setting, Download, PersonalModel, Chat, Prompt, Guide, GUIDE_STEPS, MANAGER_GUIDE_STEPS, deviceSchemas, modelSchemas, chatSchemas, promptSchemas, ToolServer, TOOL_HEALTH_STATES, toolSchemas, Provider, PROVIDER_HEALTH_STATES, providerSchemas, Skill } from '../../server/models/ai.model.js';
import { downloads, health, queue, SKILLS, DEFAULT_LIBRARY_SKILLS, COMPONENT_PAGES, COMPONENT_VALUES_PAGE, componentFaults, rankSkills, writtenToolCalls, BUILTIN_TOOLS, waitingPermissions, todayLine, SKILL_SOURCES, MAX_CHAT_SKILLS, SUGGESTED_SKILLS, setHairlineWait, setToolCallLimit, skillLinks } from '../../server/controllers/ai.controller.js';

// What a chat is sent when only the default skills are on: the Components skill, the skill map, today's date, and
// their reminders.
const COMPONENT_GUIDE = SKILLS.find((skill) => skill.id === 'components').text;
const SKILL_MAP = SKILLS.find((skill) => skill.id === 'skill-map').text;
const CLOSING_GUIDE = `REMEMBER\n${SKILLS.filter((skill) => skill.source === 'mollama' && skill.default).map((skill) => skill.remember).filter(Boolean).join('\n')}`;

const MB = 1024 * 1024;

let testStartup;
let client;

beforeAll(async () => {
    testStartup = new TestStartup('ai');
    await testStartup.initialize();
    client = testStartup.getClient();
}, 60000);

afterAll(async () => {
    resetOllamaClientFactory();
    await testStartup.cleanup();
}, 30000);

describe('Model rights', () => {
    test('MANAGE_MODELS and USE_MODELS exist', () => {
        expect(RIGHTS.MANAGE_MODELS).toBe('MANAGE_MODELS');
        expect(RIGHTS.USE_MODELS).toBe('USE_MODELS');
    });

    test.each([ROLES.OWNER, ROLES.ADMIN])('%s can manage models', (role) => {
        expect(hasRight([role], RIGHTS.MANAGE_MODELS)).toBe(true);
    });

    test.each([ROLES.SUPER_CREATOR, ROLES.CREATOR, ROLES.USER])('%s cannot manage models', (role) => {
        expect(hasRight([role], RIGHTS.MANAGE_MODELS)).toBe(false);
    });

    test.each(Object.values(ROLES))('%s can use models', (role) => {
        expect(hasRight([role], RIGHTS.USE_MODELS)).toBe(true);
    });

    test('a user with several roles gets the highest right', () => {
        expect(hasRight([ROLES.USER, ROLES.ADMIN], RIGHTS.MANAGE_MODELS)).toBe(true);
    });

    test('an unknown or missing role has neither right', () => {
        expect(hasRight(['GUEST'], RIGHTS.USE_MODELS)).toBe(false);
        expect(hasRight([], RIGHTS.MANAGE_MODELS)).toBe(false);
    });

    test('existing rights are unchanged for USER', () => {
        expect(hasRight([ROLES.USER], RIGHTS.MANAGE_OWN_ACCOUNT)).toBe(true);
        expect(hasRight([ROLES.USER], RIGHTS.MANAGE_ALL_USERS)).toBe(false);
        expect(hasRight([ROLES.USER], RIGHTS.CREATE_CONTENT)).toBe(false);
    });
});

const connectionRefusedError = () => Object.assign(new TypeError('fetch failed'), { cause: { code: 'ECONNREFUSED' } });

const simpleOllama = ({ up = true, models = ['small:1b', 'medium:7b'] } = {}) => () => ({
    version: async () => {
        if (!up) throw connectionRefusedError();
        return { version: '9.9.9-test' };
    },
    list: async () => {
        if (!up) throw connectionRefusedError();
        return { models: models.map((model) => (typeof model === 'string' ? { name: model } : model)) };
    }
});

describe('Ollama error mapping', () => {
    test.each([
        [connectionRefusedError(), 'OLLAMA_DOWN', 503],
        [{ error: "model 'x:1' not found", status_code: 404 }, 'MODEL_MISSING', 404],
        [{ error: 'something else', status_code: 404 }, 'MODEL_MISSING', 404],
        [{ error: 'model requires more system memory (9.1 GiB) than is available (4.0 GiB)' }, 'OUT_OF_MEMORY', 422],
        [{ error: 'write /models/blob: no space left on device' }, 'DISK_FULL', 422],
        [{ error: 'registry.ollama.ai/library/tiny does not support tools' }, 'NO_TOOL_SUPPORT', 422],
        [{ error: 'input exceeds the maximum context length' }, 'TOO_LONG', 422],
        [{ error: 'pull failed: unexpected EOF' }, 'DOWNLOAD_INTERRUPTED', 502],
        [{ error: 'boom' }, 'UNKNOWN', 502],
        [undefined, 'UNKNOWN', 502]
    ])('maps %o to %s', (input, code, status) => {
        const mapped = mapOllamaError(input);
        expect(mapped).toBeInstanceOf(OllamaError);
        expect(mapped.code).toBe(code);
        expect(mapped.status).toBe(status);
        expect(mapped.message.length).toBeGreaterThan(10);
        expect(mapped.action.length).toBeGreaterThan(5);
    });

    test('every code has a plain message that hides the raw error', () => {
        for (const code of ERROR_CODES) {
            const error = new OllamaError(code, 'raw internal text');
            expect(error.toJSON()).toEqual({ code, message: error.message, action: error.action });
            expect(JSON.stringify(error.toJSON())).not.toContain('raw internal text');
        }
    });

    test('an already mapped error is returned unchanged', () => {
        const original = new OllamaError('DISK_FULL');
        expect(mapOllamaError(original)).toBe(original);
    });
});

describe('Ollama client', () => {
    afterAll(() => resetOllamaClientFactory());

    test('status reports the version when Ollama answers', async () => {
        setOllamaClientFactory(simpleOllama());
        expect(await createOllamaClient('http://fake').status()).toEqual({ available: true, version: '9.9.9-test' });
    });

    test('status reports unavailable instead of throwing', async () => {
        setOllamaClientFactory(simpleOllama({ up: false }));
        expect(await createOllamaClient('http://fake').status()).toEqual({ available: false, version: null });
    });

    test('other calls throw a mapped error', async () => {
        setOllamaClientFactory(simpleOllama({ up: false }));
        await expect(createOllamaClient('http://fake').list()).rejects.toMatchObject({ code: 'OLLAMA_DOWN' });
    });
});

describe('Hardware reading', () => {
    test('parses one graphics card', () => {
        expect(parseNvidiaSmi('NVIDIA GeForce RTX 4060 Ti, 8188\r\n')).toEqual([
            { name: 'NVIDIA GeForce RTX 4060 Ti', memoryMb: 8188 }
        ]);
    });

    test('parses several graphics cards', () => {
        const cards = parseNvidiaSmi('NVIDIA RTX A6000, 49140\nNVIDIA GeForce RTX 3090, 24576\n');
        expect(cards).toHaveLength(2);
        expect(cards[1]).toEqual({ name: 'NVIDIA GeForce RTX 3090', memoryMb: 24576 });
    });

    test.each(['', null, undefined, 'not a csv line', 'Card, N/A', ', 8000'])('ignores unusable output %o', (text) => {
        expect(parseNvidiaSmi(text)).toEqual([]);
    });

    test('an Apple-chip Mac reports its built-in graphics with a share of system memory', () => {
        const [gpu] = appleGpus({ platform: 'darwin', arch: 'arm64', cpu: 'Apple M2 Pro', memoryMb: 32768 });
        expect(gpu.name).toBe('Apple M2 Pro built-in graphics');
        expect(gpu.memoryMb).toBe(21845);
        expect(appleGpus({ platform: 'darwin', arch: 'arm64', cpu: '', memoryMb: 8192 })[0].name).toBe('Apple chip built-in graphics');
    });

    test.each([
        ['an Intel Mac', { platform: 'darwin', arch: 'x64', cpu: 'Intel Core i7', memoryMb: 16384 }],
        ['a Windows computer', { platform: 'win32', arch: 'x64', cpu: 'AMD Ryzen 7', memoryMb: 32768 }],
        ['a Linux computer on an ARM chip', { platform: 'linux', arch: 'arm64', cpu: 'Cortex', memoryMb: 8192 }],
        ['a Mac with unknown memory', { platform: 'darwin', arch: 'arm64', cpu: 'Apple M1', memoryMb: 0 }]
    ])('%s gets no Apple graphics', (_label, machine) => {
        expect(appleGpus(machine)).toEqual([]);
    });

    test('Windows display adapters: an AMD card with its own memory counts, others do not', () => {
        const GB = 1024 * 1024 * 1024;
        const adapters = JSON.stringify([
            { DriverDesc: 'AMD Radeon RX 7800 XT', 'HardwareInformation.qwMemorySize': 16 * GB },
            { DriverDesc: 'AMD Radeon(TM) Graphics', 'HardwareInformation.qwMemorySize': 0.5 * GB },
            { DriverDesc: 'Intel(R) UHD Graphics 770', 'HardwareInformation.qwMemorySize': 1 * GB },
            { DriverDesc: 'NVIDIA GeForce RTX 4060', 'HardwareInformation.qwMemorySize': 8 * GB },
            { DriverDesc: 'Microsoft Basic Display Adapter' }
        ]);
        expect(parseDisplayAdapters(adapters)).toEqual([{ name: 'AMD Radeon RX 7800 XT', memoryMb: 16384 }]);
    });

    test('Windows display adapters: one adapter arrives as a single object, and bad output is ignored', () => {
        const one = JSON.stringify({ DriverDesc: 'Radeon RX 6600', 'HardwareInformation.qwMemorySize': 8 * 1024 * 1024 * 1024 });
        expect(parseDisplayAdapters(one)).toEqual([{ name: 'Radeon RX 6600', memoryMb: 8192 }]);
        for (const bad of ['', null, undefined, 'not json', '[1, 2]', '{"DriverDesc": 5}']) {
            expect(parseDisplayAdapters(bad)).toEqual([]);
        }
    });

    test('reads this machine without failing', async () => {
        const hardware = await detectHardware();
        expect(hardware.memoryMb).toBeGreaterThan(256);
        expect(hardware.cpuThreads).toBeGreaterThan(0);
        expect(typeof hardware.hostname).toBe('string');
        expect(Array.isArray(hardware.gpus)).toBe(true);
        expect(hardware.gpuMemoryMb).toBeGreaterThanOrEqual(0);
        expect(hardware.detectedAt).toBeInstanceOf(Date);
    });
});

describe('Device and settings validation', () => {
    const valid = (schema, value) => schema.validate(value).error === undefined;

    test('settings accept sensible values', () => {
        expect(valid(deviceSchemas.updateSettings, { ollamaBaseUrl: 'http://127.0.0.1:11434' })).toBe(true);
        expect(valid(deviceSchemas.updateSettings, { ollamaBaseUrl: '' })).toBe(true);
        expect(valid(deviceSchemas.updateSettings, { defaultModel: 'hf.co/unsloth/x:Q4_K_M' })).toBe(true);
        expect(valid(deviceSchemas.updateSettings, { maxConcurrentGenerations: 8, keepAliveMinutes: 0 })).toBe(true);
    });

    test.each([
        [{}],
        [{ ollamaBaseUrl: 'ftp://host' }],
        [{ ollamaBaseUrl: 'not a url' }],
        [{ defaultModel: 'bad name!' }],
        [{ maxConcurrentGenerations: 0 }],
        [{ maxConcurrentGenerations: 9 }],
        [{ keepAliveMinutes: -1 }],
        [{ unknownField: true }]
    ])('settings reject %o', (value) => {
        expect(valid(deviceSchemas.updateSettings, value)).toBe(false);
    });

    test.each([
        [{}],
        [{ name: '' }],
        [{ overrides: { memoryMb: 10 } }],
        [{ overrides: { gpuMemoryMb: -1 } }],
        [{ overrides: { somethingElse: 1 } }]
    ])('device edits reject %o', (value) => {
        expect(valid(deviceSchemas.updateDevice, value)).toBe(false);
    });
});

describe('Device, settings and setup endpoints', () => {
    let deviceId;

    beforeEach(() => {
        setOllamaClientFactory(simpleOllama());
    });

    describe('GET /api/v1/devices', () => {
        test('refuses a visitor who is not signed in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            const response = await client.get('/api/v1/devices');
            expect(response.status).toBe(401);
        });

        test('shows this computer to a regular user, without manage rights', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices');

            expect(response.status).toBe(200);
            expect(response.data.devices).toHaveLength(1);
            expect(response.data.permissions.manage).toBe(false);

            const [device] = response.data.devices;
            deviceId = device.id;
            expect(device.kind).toBe('local');
            expect(device.service).toBe('ai-base');
            expect(device.online).toBe(true);
            expect(device.hardware.memoryMb).toBeGreaterThan(256);
            expect(typeof device.hardware.hasGpu).toBe('boolean');
            expect(device.edited).toBe(false);
            expect(device.ollama).toEqual({ available: true, version: '9.9.9-test', modelCount: 2, chatCount: 2 });
        });

        test('gives manage rights to admin and owner', async () => {
            for (const role of ['admin', 'owner']) {
                await testStartup.loginAsUser(role);
                const response = await client.get('/api/v1/devices');
                expect(response.status).toBe(200);
                expect(response.data.permissions.manage).toBe(true);
            }
        });

        test('still answers when Ollama is not running', async () => {
            setOllamaClientFactory(simpleOllama({ up: false }));
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices');

            expect(response.status).toBe(200);
            expect(response.data.devices[0].ollama).toEqual({ available: false, version: null, modelCount: 0, chatCount: 0 });
        });

        test('models that only make embeddings are counted, but not as models that can chat', async () => {
            setOllamaClientFactory(simpleOllama({ models: [{ name: 'bge-large:335m', details: { family: 'bert' } }, { name: 'mxbai-embed-large:335m', details: { family: 'bert' } }] }));
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices');
            expect(response.data.devices[0].ollama).toMatchObject({ available: true, modelCount: 2, chatCount: 0 });

            setOllamaClientFactory(simpleOllama({ models: ['mxbai-embed-large:335m', 'nomic-embed-text:latest', 'small:1b'] }));
            const mixed = await client.get('/api/v1/devices');
            expect(mixed.data.devices[0].ollama).toMatchObject({ modelCount: 3, chatCount: 1 });
        });

        test('is never cached', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices');
            expect(String(response.headers['cache-control'] || '')).toMatch(/no-store|no-cache/);
        });

        test('stores the device once, marked with the service', async () => {
            await client.get('/api/v1/devices');
            const stored = await Device.find({ kind: 'local' });
            expect(stored).toHaveLength(1);
            expect(stored[0].service).toBe('ai-base');
            expect(stored[0].hardware.memoryMb).toBeGreaterThan(256);
        });
    });

    describe('GET /api/v1/devices/:id', () => {
        test('returns the device', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get(`/api/v1/devices/${deviceId}`);
            expect(response.status).toBe(200);
            expect(response.data.device.id).toBe(deviceId);
        });

        test('rejects an id that is not a database id', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices/not-an-id');
            expect(response.status).toBe(400);
        });

        test('answers 404 for an unknown device', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/devices/64b7f0c2a1b2c3d4e5f60718');
            expect(response.status).toBe(404);
        });
    });

    describe('PUT /api/v1/devices/:id', () => {
        test('refuses a regular user', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.put(`/api/v1/devices/${deviceId}`, { name: 'Renamed' });
            expect(response.status).toBe(403);
        });

        test('lets an admin rename the device and correct its hardware', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put(`/api/v1/devices/${deviceId}`, {
                name: 'Studio PC',
                overrides: { memoryMb: 65536, gpuName: 'Test Card', gpuMemoryMb: 24576 }
            });

            expect(response.status).toBe(200);
            expect(response.data.device.name).toBe('Studio PC');
            expect(response.data.device.edited).toBe(true);
            expect(response.data.device.hardware).toMatchObject({
                memoryMb: 65536, gpuName: 'Test Card', gpuMemoryMb: 24576, hasGpu: true
            });
        });

        test('keeps the correction after a new detection', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.post(`/api/v1/devices/${deviceId}/detect`, {});
            expect(response.status).toBe(200);
            expect(response.data.device.hardware.gpuMemoryMb).toBe(24576);
            expect(response.data.device.detected.memoryMb).not.toBe(65536);
        });

        test('goes back to detected values when the correction is cleared', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put(`/api/v1/devices/${deviceId}`, {
                overrides: { memoryMb: null, gpuName: '', gpuMemoryMb: null }
            });

            expect(response.status).toBe(200);
            expect(response.data.device.edited).toBe(false);
            expect(response.data.device.hardware.memoryMb).toBe(response.data.device.detected.memoryMb);
        });

        test('rejects values outside the allowed range', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put(`/api/v1/devices/${deviceId}`, { overrides: { memoryMb: 10 } });
            expect(response.status).toBe(400);
        });

        test('rejects an empty change', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put(`/api/v1/devices/${deviceId}`, {});
            expect(response.status).toBe(400);
        });
    });

    describe('POST /api/v1/devices/:id/detect', () => {
        test('refuses a regular user', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.post(`/api/v1/devices/${deviceId}/detect`, {});
            expect(response.status).toBe(403);
        });
    });

    describe('Settings', () => {
        test('a regular user can read settings but not change them', async () => {
            await testStartup.loginAsUser('user');
            const read = await client.get('/api/v1/settings');
            expect(read.status).toBe(200);
            expect(read.data.permissions.manage).toBe(false);
            expect(read.data.settings.ollamaBaseUrl).toBe(process.env.OLLAMA_BASE_URL);
            expect(read.data.settings.ollamaBaseUrlIsDefault).toBe(true);
            expect(read.data.settings.maxConcurrentGenerations).toBe(1);

            const write = await client.put('/api/v1/settings', { keepAliveMinutes: 10 });
            expect(write.status).toBe(403);
        });

        test('an admin can change settings and they take effect at once', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put('/api/v1/settings', {
                ollamaBaseUrl: 'http://192.168.50.10:11434',
                defaultModel: 'medium:7b',
                maxConcurrentGenerations: 2,
                keepAliveMinutes: 30
            });

            expect(response.status).toBe(200);
            expect(response.data.settings).toMatchObject({
                ollamaBaseUrl: 'http://192.168.50.10:11434',
                ollamaBaseUrlIsDefault: false,
                defaultModel: 'medium:7b',
                maxConcurrentGenerations: 2,
                keepAliveMinutes: 30
            });

            const read = await client.get('/api/v1/settings');
            expect(read.data.settings.ollamaBaseUrl).toBe('http://192.168.50.10:11434');
        });

        test('clearing the address goes back to the default', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.put('/api/v1/settings', { ollamaBaseUrl: '' });
            expect(response.status).toBe(200);
            expect(response.data.settings.ollamaBaseUrl).toBe(process.env.OLLAMA_BASE_URL);
            expect(response.data.settings.ollamaBaseUrlIsDefault).toBe(true);
        });

        test('rejects a bad address and out-of-range numbers', async () => {
            await testStartup.loginAsUser('admin');
            expect((await client.put('/api/v1/settings', { ollamaBaseUrl: 'ftp://host' })).status).toBe(400);
            expect((await client.put('/api/v1/settings', { maxConcurrentGenerations: 0 })).status).toBe(400);
            expect((await client.put('/api/v1/settings', {})).status).toBe(400);
        });

        test('keeps one settings record, marked with the service', async () => {
            const stored = await Setting.find({});
            expect(stored).toHaveLength(1);
            expect(stored[0].service).toBe('ai-base');
        });
    });

    describe('POST /api/v1/settings/test-connection', () => {
        test('refuses a regular user', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.post('/api/v1/settings/test-connection', { ollamaBaseUrl: 'http://127.0.0.1:11434' });
            expect(response.status).toBe(403);
        });

        test('reports success with the version', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.post('/api/v1/settings/test-connection', { ollamaBaseUrl: 'http://127.0.0.1:11434' });
            expect(response.status).toBe(200);
            expect(response.data.ollama).toEqual({ available: true, version: '9.9.9-test' });
        });

        test('reports a plain error when nothing answers', async () => {
            setOllamaClientFactory(simpleOllama({ up: false }));
            await testStartup.loginAsUser('admin');
            const response = await client.post('/api/v1/settings/test-connection', { ollamaBaseUrl: 'http://127.0.0.1:11434' });
            expect(response.status).toBe(503);
            expect(response.data).toMatchObject({ success: false, code: 'OLLAMA_DOWN' });
            expect(response.data.action).toBeDefined();
        });

        test('does not save the address it tested', async () => {
            await testStartup.loginAsUser('admin');
            await client.post('/api/v1/settings/test-connection', { ollamaBaseUrl: 'http://10.9.8.7:11434' });
            const read = await client.get('/api/v1/settings');
            expect(read.data.settings.ollamaBaseUrl).not.toBe('http://10.9.8.7:11434');
        });

        test('rejects a missing or malformed address', async () => {
            await testStartup.loginAsUser('admin');
            expect((await client.post('/api/v1/settings/test-connection', {})).status).toBe(400);
            expect((await client.post('/api/v1/settings/test-connection', { ollamaBaseUrl: 'nope' })).status).toBe(400);
        });
    });

    describe('GET /api/v1/setup/status', () => {
        test('refuses a visitor who is not signed in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            const response = await client.get('/api/v1/setup/status');
            expect(response.status).toBe(401);
        });

        test('reports every step as ready when Ollama has models', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/setup/status');
            expect(response.status).toBe(200);
            expect(response.data.steps).toEqual({ ollama: true, device: true, model: true });
            expect(response.data.ollama.version).toBe('9.9.9-test');
        });

        test('reports no model yet when Ollama is empty', async () => {
            setOllamaClientFactory(simpleOllama({ models: [] }));
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/setup/status');
            expect(response.data.steps).toEqual({ ollama: true, device: true, model: false });
        });

        test('explains in plain words when Ollama is not running', async () => {
            setOllamaClientFactory(simpleOllama({ up: false }));
            await testStartup.loginAsUser('user');
            const response = await client.get('/api/v1/setup/status');

            expect(response.status).toBe(200);
            expect(response.data.steps.ollama).toBe(false);
            expect(response.data.ollama).toMatchObject({ available: false, code: 'OLLAMA_DOWN' });
            expect(response.data.ollama.message).toMatch(/not running/i);
            expect(response.data.ollama.action).toBeDefined();
        });
    });
});

// Sample models for the fit and ranking tests. The app itself reads the live list from ollama.com.
const catalogue = [
    {"tag":"llama3.2:3b","name":"Llama 3.2 3B","description":"Small and quick. Good for short questions, rewording and simple summaries.","goodFor":["chat","writing"],"parameters":3.2,"sizeMb":2000,"moe":false,"capabilities":["tools"]},
    {"tag":"phi4-mini:3.8b","name":"Phi-4 Mini","description":"A compact model that reasons well for its size. Good for quick answers on modest hardware.","goodFor":["chat","reasoning"],"parameters":3.8,"sizeMb":2500,"moe":false,"capabilities":["tools"]},
    {"tag":"gemma3:4b","name":"Gemma 3 4B","description":"A small general model with a friendly writing style. Good for drafting and everyday chat.","goodFor":["chat","writing"],"parameters":4.3,"sizeMb":3300,"moe":false,"capabilities":["vision"]},
    {"tag":"mistral:7b","name":"Mistral 7B","description":"A dependable all-rounder for chat, summaries and short documents.","goodFor":["chat","writing"],"parameters":7.2,"sizeMb":4400,"moe":false,"capabilities":["tools"]},
    {"tag":"qwen2.5-coder:7b","name":"Qwen 2.5 Coder 7B","description":"Tuned for programming. Good for explaining, writing and fixing code.","goodFor":["code"],"parameters":7.6,"sizeMb":4700,"moe":false,"capabilities":["tools"]},
    {"tag":"llama3.1:8b","name":"Llama 3.1 8B","description":"A well-rounded everyday assistant that can also use tools, so it suits simple agents.","goodFor":["chat","writing"],"parameters":8,"sizeMb":4900,"moe":false,"capabilities":["tools"]},
    {"tag":"deepseek-r1:8b","name":"DeepSeek R1 8B","description":"Thinks step by step before answering. Good for problems that need working out.","goodFor":["reasoning"],"parameters":8.2,"sizeMb":5200,"moe":false,"capabilities":["thinking"]},
    {"tag":"gemma3:12b","name":"Gemma 3 12B","description":"A stronger writer than the small models. Good for longer drafts and careful rewording.","goodFor":["chat","writing"],"parameters":12.2,"sizeMb":8100,"moe":false,"capabilities":["vision"]},
    {"tag":"qwen2.5:14b","name":"Qwen 2.5 14B","description":"A capable general model for documents, analysis and tool use.","goodFor":["chat","writing","reasoning"],"parameters":14.8,"sizeMb":9000,"moe":false,"capabilities":["tools"]},
    {"tag":"phi4:14b","name":"Phi-4 14B","description":"Careful and precise. Good for analysis, structured answers and checking work.","goodFor":["reasoning","writing"],"parameters":14.7,"sizeMb":9100,"moe":false,"capabilities":[]},
    {"tag":"deepseek-r1:14b","name":"DeepSeek R1 14B","description":"Step-by-step reasoning with more depth than the 8B version.","goodFor":["reasoning"],"parameters":14.8,"sizeMb":9000,"moe":false,"capabilities":["thinking"]},
    {"tag":"gpt-oss:20b","name":"GPT-OSS 20B","description":"A strong general model that reasons and uses tools. Only part of it is active at a time, so it runs on less graphics memory than its size suggests.","goodFor":["chat","reasoning","writing"],"parameters":20.9,"sizeMb":14000,"moe":true,"capabilities":["tools","thinking"]},
    {"tag":"gemma3:27b","name":"Gemma 3 27B","description":"The largest Gemma. Good for long-form writing and detailed answers on a large graphics card.","goodFor":["chat","writing"],"parameters":27.4,"sizeMb":17000,"moe":false,"capabilities":["vision"]},
    {"tag":"qwen3:30b","name":"Qwen 3 30B","description":"A large general model that thinks before answering and uses tools. Only part of it is active at a time, so system memory can carry most of it.","goodFor":["chat","reasoning","writing"],"parameters":30.5,"sizeMb":19000,"moe":true,"capabilities":["tools","thinking"]},
    {"tag":"qwen3-coder:30b","name":"Qwen 3 Coder 30B","description":"A large programming model for bigger coding tasks and agents. Only part of it is active at a time, so system memory can carry most of it.","goodFor":["code"],"parameters":30.5,"sizeMb":19000,"moe":true,"capabilities":["tools"]},
    {"tag":"qwen2.5-coder:32b","name":"Qwen 2.5 Coder 32B","description":"A top programming model for a 24 GB graphics card.","goodFor":["code"],"parameters":32.8,"sizeMb":20000,"moe":false,"capabilities":["tools"]},
    {"tag":"llama3.3:70b","name":"Llama 3.3 70B","description":"A very large general model for workstations with a lot of graphics memory.","goodFor":["chat","writing","reasoning"],"parameters":70.6,"sizeMb":43000,"moe":false,"capabilities":["tools"]}
];

// Stands in for ollama.com: a search page built from the sample models, and a tags page with each size on disk.
const library = { down: false, empty: false, asked: [], marked: [] };
const libraryNames = () => [...new Set(catalogue.map((model) => model.tag.split(':')[0]))];
const fakeLibrary = async (address, options = {}) => {
    const url = new URL(address);
    library.asked.push(url.pathname + url.search);
    if (options.headers?.['HX-Request']) library.marked.push(url.pathname + url.search);
    if (library.down) throw new TypeError('fetch failed');
    const page = (html) => ({ ok: true, status: 200, text: async () => html });
    const tags = url.pathname.match(/^\/library\/([^/]+)\/tags$/);
    if (tags) {
        return page(catalogue.filter((model) => model.tag.startsWith(`${tags[1]}:`)).map((model) => `<a href="/library/${model.tag}"><span>${model.tag}</span><span>abc123 • ${model.sizeMb}MB • 128K context window</span></a>`).join(''));
    }
    if (library.empty) return page('<html><body>Nothing here</body></html>');
    const wanted = (url.searchParams.get('q') || '').toLowerCase();
    const second = url.searchParams.get('page') === '2' && Boolean(options.headers?.['HX-Request']);
    if (Number(url.searchParams.get('page')) > 2) return page('<ul></ul>');
    const names = libraryNames().filter((name) => name.includes(wanted)).filter((name, index) => (wanted ? true : second ? index >= 8 : index < 8));
    const cloud = !second && 'kimi-k3'.includes(wanted) ? '<li><a href="/library/kimi-k3"><h2><span>kimi-k3</span></h2><span>tools</span><span>cloud</span><p>Runs on the maker\'s servers.</p></a></li>' : '';
    const items = names.map((name) => {
        const sizes = catalogue.filter((model) => model.tag.startsWith(`${name}:`));
        return `<li><a href="/library/${name}" class="group"><h2 title="${name}"><span>${name}</span></h2><span title="Updated October 1, 2026">1mo</span>${sizes[0].capabilities.map((ability) => `<span class="a">${ability}</span>`).join('')}${sizes.map((model) => `<span class="s">${model.tag.split(':')[1]}</span>`).join('')}<p class="d">${sizes[0].description}${sizes[0].moe ? ' A mixture of experts model.' : ''}</p><span title="1,234,567 downloads"><span>1.2M</span></span></a></li>`;
    });
    return page(`<ul>${cloud}${items.join('')}</ul>${!wanted && !second ? '<div hx-get="/search?page=2"></div>' : ''}`);
};
setLibraryFetch(fakeLibrary);
const entry = (tag) => catalogue.find((model) => model.tag === tag);

const notFound = (name) => Object.assign(new Error(`model '${name}' not found`), { error: `model '${name}' not found`, status_code: 404 });
const refused = () => Object.assign(new TypeError('fetch failed'), { cause: { code: 'ECONNREFUSED' } });

const installed = (name, sizeMb = 100) => ({
    name,
    size: sizeMb * MB,
    modified_at: '2026-11-01T10:00:00Z',
    details: { family: 'testfam', parameter_size: '7.6B', quantization_level: 'Q4_K_M' }
});

const state = {};
const resetState = () => {
    Object.assign(state, {
        up: true,
        models: [installed('qwen2.5-coder:7b', 4700), installed('custom-thing:latest', 50)],
        running: [],
        unloaded: [],
        created: [],
        pullSizeMb: 10,
        pullFailsMidway: false,
        hold: null,
        stall: false,
        silent: false,
        chatReply: 'A short answer',
        chatCalls: [],
        guidesSeen: 0,
        valuesSeen: 0,
        toolPlan: [],
        toolsAsText: false,
        plan: { searches: [] },
        planCalls: [],
        planAborted: 0,
        searchRequests: [],
        reviews: [],
        reviewCalls: [],
        freshResults: false,
        enginesLimited: false,
        junkResults: false,
        planHold: false,
        searches: [],
        fetches: [],
        searchFails: false,
        searchStatus: null,
        searchUrls: [],
        searchGuides: 0,
        noTools: false,
        chatAborted: 0,
        chatHold: false,
        chatError: null,
        streamErrors: [],
        chatFailsMidway: false,
        thinkingModel: false,
        contextLength: 32768,
        filesDown: false
    });
};

const fakeChat = (request) => {
    let aborted = false;
    let wake = () => {};
    const check = () => {
        if (aborted) throw Object.assign(new Error('The operation was aborted'), { name: 'AbortError' });
    };
    const pause = (ms) => new Promise((resolve) => {
        wake = resolve;
        setTimeout(resolve, ms);
    });
    // A test may line up several replies, given one after another; otherwise every reply is the same.
    const words = (state.replyPlan?.length > 0 ? state.replyPlan.shift() : state.chatReply).split(' ');

    return {
        abort() {
            if (!aborted) state.chatAborted += 1;
            aborted = true;
            wake();
        },
        async *[Symbol.asyncIterator]() {
            if (request.think) yield { message: { role: 'assistant', content: '', thinking: 'Let me consider this.' } };
            // A test may make replies break off partway with what Ollama said, one after another; an empty entry is a reply that does not break.
            const broken = state.streamErrors?.length > 0 ? state.streamErrors.shift() : null;
            for (let index = 0; index < words.length; index++) {
                if (broken && index === 1) throw Object.assign(new Error(broken), { error: broken });
                await pause(2);
                check();
                yield { message: { role: 'assistant', content: index < words.length - 1 ? `${words[index]} ` : words[index] } };
                if (state.chatFailsMidway) throw refused();
                while (state.chatHold) {
                    await pause(10);
                    check();
                }
            }
            yield {
                message: { role: 'assistant', content: '' },
                done: true,
                // A test may say how replies end, one after another: "length" is a model that ran out of room.
                ...(state.endings?.length > 0 ? { done_reason: state.endings.shift() } : {}),
                eval_count: words.length,
                prompt_eval_count: 20,
                total_duration: 2e9,
                eval_duration: 1e9
            };
        }
    };
};

const fakePull = (model) => {
    let aborted = false;
    let wake = () => {};
    const total = state.pullSizeMb * MB;
    const check = () => {
        if (aborted) throw Object.assign(new Error('The operation was aborted'), { name: 'AbortError' });
    };
    const pause = (ms) => new Promise((resolve) => {
        wake = resolve;
        setTimeout(resolve, ms);
    });

    return {
        abort() {
            aborted = true;
            wake();
        },
        async *[Symbol.asyncIterator]() {
            yield { status: 'pulling manifest' };
            while (state.silent) {
                await pause(20);
                check();
            }
            await pause(5);
            check();
            yield { status: 'pulling layer', digest: 'sha256:one', total, completed: total / 2 };
            if (state.pullFailsMidway) throw new Error('unexpected EOF');
            while (state.stall) {
                await pause(20);
                check();
                yield { status: 'pulling layer', digest: 'sha256:one', total, completed: total / 2 };
            }
            while (state.hold) {
                await pause(20);
                check();
            }
            await pause(5);
            check();
            yield { status: 'pulling layer', digest: 'sha256:one', total, completed: total };
            yield { status: 'success' };
            if (!state.models.some((existing) => existing.name === model)) {
                state.models.push(installed(model, state.pullSizeMb));
            }
        }
    };
};

const EVERY_CHAT_TOOLS = ['open_component', 'find_skill', 'ask_user', 'save_note', 'list_my_files', 'read_my_file', 'save_to_files', 'date_time', 'calculator', 'read_web_page'];
const DEFAULT_SKILL_NAMES = SKILLS.filter((skill) => DEFAULT_LIBRARY_SKILLS[skill.id]).map((skill) => skill.name);
const fakeOllama = (host, signal) => ({
    version: async () => {
        if (!state.up) throw refused();
        return { version: '9.9.9-test' };
    },
    list: async () => {
        if (!state.up) throw refused();
        return { models: state.models };
    },
    ps: async () => {
        if (!state.up) throw refused();
        return { models: state.running };
    },
    show: async ({ model }) => {
        if (!state.models.some((existing) => existing.name === model)) throw notFound(model);
        return {
            modelfile: `FROM ${model}`,
            parameters: 'temperature 0.7',
            template: '{{ .Prompt }}',
            capabilities: [...(state.noTools ? ['completion'] : state.thinkingModel ? ['completion', 'tools', 'thinking'] : ['completion', 'tools']), ...(state.vision ? ['vision'] : [])],
            model_info: { 'testfam.context_length': state.contextLength },
            details: { family: 'testfam' }
        };
    },
    delete: async ({ model }) => {
        const before = state.models.length;
        state.models = state.models.filter((existing) => existing.name !== model);
        if (state.models.length === before) throw notFound(model);
        return { status: 'success' };
    },
    create: async (request) => {
        state.created.push(request);
        state.models.push(installed(`${request.model}:latest`, 1));
        return { status: 'success' };
    },
    generate: async ({ model, keep_alive, options }) => {
        if (!state.models.some((existing) => existing.name === model)) throw notFound(model);
        state.unloaded.push(options ? { model, keep_alive, num_ctx: options.num_ctx } : { model, keep_alive });
        return { done: true };
    },
    chat: async (request) => {
        if (!state.up) throw refused();
        if (!state.models.some((existing) => existing.name === request.model)) throw notFound(request.model);
        // Folding a long chat's memory runs in the background after a reply; it is kept apart from the chat's own calls.
        if (request.stream === false && String(request.messages?.[0]?.content || '').startsWith('Rewrite the memory')) {
            (state.memoryCalls ||= []).push(request);
            return { message: { role: 'assistant', content: state.memoryReply || '- The user asked about ducks.' }, done: true };
        }
        const lead = request.messages[0];
        // The default skill, the date line and the skill's reminder are taken off, so a test sees only what a chat adds.
        if (lead?.role === 'system') {
            let own = lead.content;
            if (own.startsWith(COMPONENT_GUIDE)) {
                state.guidesSeen += 1;
                own = own.slice(COMPONENT_GUIDE.length).replace(/^\n\n/, '');
            }
            if (own.startsWith(COMPONENT_VALUES_PAGE)) {
                state.valuesSeen += 1;
                own = own.slice(COMPONENT_VALUES_PAGE.length).replace(/^\n\n/, '');
            }
            if (own.startsWith(SKILL_MAP)) {
                state.mapsSeen = (state.mapsSeen || 0) + 1;
                own = own.slice(SKILL_MAP.length).replace(/^\n\n/, '');
            }
            if (own.endsWith(CLOSING_GUIDE)) own = own.slice(0, own.length - CLOSING_GUIDE.length).replace(/\n\n$/, '');
            own = own.split('\n\n').filter((part) => part !== todayLine()).join('\n\n');
            request = { ...request, messages: own ? [{ role: 'system', content: own }, ...request.messages.slice(1)] : request.messages.slice(1) };
        }
        // The search of the skill library and the built-in tools are offered with every chat; like the default skill
        // they are taken off here, so a test sees only the tools a chat adds. What was really offered is kept in
        // state.toolsSeen.
        const offered = request.tools || null;
        (state.toolsSeen ||= []).push(offered);
        // A skill that is on by default for models that can use tools is listed in every such chat; a list of nothing
        // but those is taken off too.
        // The skill map's worked examples are offered with every chat too, as files of a skill that is always on.
        const everyChat = (tool) => EVERY_CHAT_TOOLS.includes(tool.function?.name)
            || (tool.function?.name === 'open_skill' && tool.function.parameters.properties.skill.enum.every((name) => DEFAULT_SKILL_NAMES.includes(name)))
            || (tool.function?.name === 'open_skill_file' && tool.function.parameters.properties.skill.enum.every((id) => id === 'skill-map'));
        if (offered?.some(everyChat)) {
            const rest = offered.filter((tool) => !everyChat(tool));
            request = { ...request, tools: rest.length > 0 ? rest : undefined };
        }
        state.chatCalls.push(request);
        // A model still loading: Ollama has not answered, so only the request's own signal can end it.
        while (state.loadHold && !signal?.aborted) await new Promise((resolve) => setTimeout(resolve, 10));
        if (state.loadHold && signal?.aborted) {
            state.loadAborted += 1;
            throw Object.assign(new Error('This operation was aborted'), { name: 'AbortError' });
        }
        if (state.chatError) throw Object.assign(new Error(state.chatError), { error: state.chatError });
        if (request.format) {
            const answer = (content) => {
                const result = { message: { role: 'assistant', content }, done: true };
                if (request.stream === false) return result;
                let aborted = false;
                return {
                    abort() {
                        aborted = true;
                        state.planAborted += 1;
                    },
                    async *[Symbol.asyncIterator]() {
                        while (state.planHold && !aborted) await new Promise((resolve) => setTimeout(resolve, 10));
                        if (aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' });
                        yield result;
                    }
                };
            };
            state.chatCalls.pop();
            if (request.format.properties?.enough) {
                state.reviewCalls.push(request);
                const review = state.reviews.shift() ?? { enough: true, searches: [], pages: [] };
                return answer(typeof review === 'string' ? review : JSON.stringify(review));
            }
            state.planCalls.push(request);
            return answer(typeof state.plan === 'string' ? state.plan : JSON.stringify(state.plan));
        }
        if (request.stream === false) return { message: { role: 'assistant', content: state.chatReply }, done: true };
        if (offered && state.toolPlan.length > 0) {
            const calls = state.toolPlan.shift().map(([name, args]) => ({ function: { name, arguments: args } }));
            if (state.toolsAsText) {
                const text = calls.map((call) => JSON.stringify({ name: call.function.name, arguments: call.function.arguments })).join('\n');
                return {
                    abort() {},
                    async *[Symbol.asyncIterator]() {
                        yield { message: { role: 'assistant', content: text.slice(0, 20) } };
                        yield { message: { role: 'assistant', content: text.slice(20) } };
                        yield { message: { role: 'assistant', content: '' }, done: true, eval_count: 3, prompt_eval_count: 20, total_duration: 2e8, eval_duration: 1e8 };
                    }
                };
            }
            return {
                abort() {},
                async *[Symbol.asyncIterator]() {
                    yield { message: { role: 'assistant', content: '', tool_calls: calls } };
                    yield { message: { role: 'assistant', content: '' }, done: true, eval_count: 3, prompt_eval_count: 20, total_duration: 2e8, eval_duration: 1e8 };
                }
            };
        }
        return fakeChat(request);
    },
    pull: async ({ model }) => {
        if (!state.up) throw refused();
        if (model.startsWith('nosuch')) {
            throw Object.assign(new Error('pull model manifest: file does not exist'), { error: 'pull model manifest: file does not exist' });
        }
        return fakePull(model);
    }
});

describe('Fit rules', () => {
    const laptop = { memoryMb: 32768, gpuMemoryMb: 8192 };
    const workstation = { memoryMb: 65536, gpuMemoryMb: 24576 };
    const noCard = { memoryMb: 16384, gpuMemoryMb: 0 };

    test('a model too large for one computer fits several pooled when what they can give together is enough', () => {
        // Each gives its graphics memory less the headroom, or with no graphics card its system memory less the reserve.
        expect(poolShareMb(laptop)).toBe(8192 - 1536);
        expect(poolShareMb(noCard)).toBe(16384 - 4096);
        const big = { sizeMb: 15000, parameters: 24 };
        expect(fitModel(big, noCard).fits).toBe(false);
        expect(fitPool(big, [noCard, noCard])).toEqual({ fits: true, memoryMb: 12288 * 2, memoryNeededMb: memoryNeededMb(big) });
        // A computer with a graphics card gives only its graphics memory to a pool.
        expect(fitPool(big, [laptop, laptop]).fits).toBe(false);
        // One computer is not a pool, and a model that only runs online is never pooled.
        expect(fitPool(big, [workstation])).toBeNull();
        expect(fitPool({ ...big, cloudOnly: true }, [laptop, noCard])).toBeNull();
    });

    test('a pooled model fills the fastest device first', () => {
        const mac = { memoryMb: 18432, gpuMemoryMb: 12288, gpuShared: true };
        const pc = { memoryMb: 32768, gpuMemoryMb: 8192 };
        // Order is llama.cpp's: the lending Mac first, then the main computer's graphics card, which is faster and fills first.
        expect(splitBySpeed([mac, pc], 15000)).toEqual([8.1, 6.5]);
        // A model too big for both together is split by what each can give.
        expect(splitBySpeed([mac, pc], 40000)).toEqual([10.5, 6.5]);
    });

    test('8 GB of graphics memory runs 7 to 9B models fully on the GPU', () => {
        for (const tag of ['mistral:7b', 'qwen2.5-coder:7b', 'llama3.1:8b']) {
            expect(fitModel(entry(tag), laptop).verdict).toBe('gpu');
        }
    });

    test('8 GB plus 32 GB of system memory runs a 30B mixture-of-experts model with CPU help', () => {
        const fit = fitModel(entry('qwen3-coder:30b'), laptop);
        expect(fit.verdict).toBe('mixed');
        expect(fit.label).toBe(VERDICTS.mixed);
        expect(fit.tokensPerSecond[0]).toBeGreaterThanOrEqual(15);
    });

    test('a 70B model is too large for that machine', () => {
        const fit = fitModel(entry('llama3.3:70b'), laptop);
        expect(fit.verdict).toBe('too_large');
        expect(fit.fits).toBe(false);
        expect(fit.tokensPerSecond).toBeNull();
    });

    test('24 GB of graphics memory runs a 32B model fully on the GPU', () => {
        expect(fitModel(entry('qwen2.5-coder:32b'), workstation).verdict).toBe('gpu');
    });

    test('a machine without a graphics card runs small models on the processor', () => {
        expect(fitModel(entry('llama3.2:3b'), noCard).verdict).toBe('cpu');
        expect(fitModel(entry('qwen3:30b'), noCard).verdict).toBe('too_large');
    });

    test('headroom is kept for desktop apps', () => {
        const model = { sizeMb: 5000, parameters: 8, moe: false, goodFor: ['chat'] };
        const needed = memoryNeededMb(model);
        expect(fitModel(model, { memoryMb: 0, gpuMemoryMb: needed + 1536 }).verdict).toBe('gpu');
        expect(fitModel(model, { memoryMb: 0, gpuMemoryMb: needed + 1535 }).verdict).toBe('too_large');
    });

    test('missing or odd hardware never throws', () => {
        for (const hardware of [undefined, null, {}, { memoryMb: 'abc', gpuMemoryMb: -5 }]) {
            expect(fitModel(entry('llama3.2:3b'), hardware).verdict).toBe('too_large');
        }
    });

    test('three different models are recommended first', () => {
        const ranked = rankCatalogue(catalogue, laptop);
        expect(ranked.slice(0, 3).map((model) => model.pick)).toEqual(['fast', 'balanced', 'best']);
        expect(new Set(ranked.slice(0, 3).map((model) => model.tag)).size).toBe(3);
        expect([null, 'code', 'reasoning']).toContain(ranked[3].pick);
        expect(new Set(ranked.slice(0, 4).map((model) => model.tag)).size).toBe(4);
        expect(ranked.slice(4).every((model) => model.pick === null)).toBe(true);
        expect(ranked.find((model) => model.pick === 'best').fit.fits).toBe(true);
        for (const model of ranked.filter((entry) => entry.pick)) {
            expect(model.pickLabel.length).toBeGreaterThan(3);
            expect(model.pickWhy).toMatch(/\d+(\.\d)? GB/);
            expect(model.pickWhy).toMatch(/tokens a second|speed unknown/);
        }
        const fastest = ranked.find((model) => model.pick === 'fast');
        const capable = ranked.filter((model) => model.parameters >= 1 && ['gpu', 'cpu'].includes(model.fit.verdict));
        if (capable.length > 0) expect(fastest.parameters).toBeGreaterThanOrEqual(1);
    });

    test('a very small machine gets no recommendation instead of a wrong one', () => {
        const ranked = rankCatalogue(catalogue, { memoryMb: 4096, gpuMemoryMb: 0 });
        expect(ranked.some((model) => model.pick)).toBe(false);
        expect(ranked).toHaveLength(catalogue.length);
    });

    test('the catalogue covers every size tier with complete entries', () => {
        expect(catalogue.length).toBeGreaterThanOrEqual(12);
        const sizes = catalogue.map((model) => model.parameters);
        expect(sizes.some((size) => size <= 4.5)).toBe(true);
        expect(sizes.some((size) => size > 4.5 && size <= 9)).toBe(true);
        expect(sizes.some((size) => size > 9 && size <= 15)).toBe(true);
        expect(catalogue.some((model) => model.moe && model.parameters >= 30)).toBe(true);
        expect(sizes.some((size) => size >= 70)).toBe(true);
        for (const model of catalogue) {
            expect(modelSchemas.named.validate({ name: model.tag }).error).toBeUndefined();
            expect(model.name && model.description).toBeTruthy();
            expect(model.goodFor.length).toBeGreaterThan(0);
            expect(model.sizeMb).toBeGreaterThan(0);
            expect(Array.isArray(model.capabilities)).toBe(true);
        }
        expect(new Set(catalogue.map((model) => model.tag)).size).toBe(catalogue.length);
    });
});

describe('Model validation', () => {
    const valid = (schema, value) => schema.validate(value).error === undefined;

    test.each(['llama3.1:8b', 'hf.co/unsloth/x:Q4_K_M', 'a'])('accepts the model name %s', (name) => {
        expect(valid(modelSchemas.named, { name })).toBe(true);
    });

    test.each(['', 'bad name', 'semi;colon', 'x'.repeat(201), 'quote"'])('rejects the model name %o', (name) => {
        expect(valid(modelSchemas.named, { name })).toBe(false);
    });

    test('personal models need a simple name, a display name and a base', () => {
        expect(valid(modelSchemas.create, { name: 'careful-reviewer', displayName: 'Careful reviewer', from: 'llama3.1:8b' })).toBe(true);
        expect(valid(modelSchemas.create, { name: 'Careful', displayName: 'x', from: 'llama3.1:8b' })).toBe(false);
        expect(valid(modelSchemas.create, { name: 'has/slash', displayName: 'x', from: 'llama3.1:8b' })).toBe(false);
        expect(valid(modelSchemas.create, { name: 'ok', displayName: 'x' })).toBe(false);
        expect(valid(modelSchemas.create, { name: 'ok', displayName: 'x', from: 'a', system: 'y'.repeat(20001) })).toBe(false);
        expect(valid(modelSchemas.create, { name: 'ok', displayName: 'x', from: 'a', parameters: { temperature: 0.2, stop: 'END', raw: true } })).toBe(true);
        expect(valid(modelSchemas.create, { name: 'ok', displayName: 'x', from: 'a', parameters: { nested: { no: 1 } } })).toBe(false);
    });
});

describe('Model endpoints', () => {
    // A removal or download names the computer it is for, as the page does; here, the suite's stand-in computer.
    const CONFIRMED = ['/models/delete', '/models/pull'];
    let standIn = null;
    const post = async (path, body) => {
        if (!CONFIRMED.includes(path)) return client.post(`/api/v1${path}`, body);
        standIn = standIn || String((await Device.findOne({ kind: 'local', service: 'ai-base' }))._id);
        return client.post(`/api/v1${path}`, body, { headers: { 'X-Mollama-Device': standIn } });
    };
    const get = (path) => client.get(`/api/v1${path}`);
    const signOut = () => {
        client.clearCookies();
        client.authenticatedAs = null;
    };

    const BIG_DEVICE = { memoryMb: 262144, gpuMemoryMb: 98304, gpuName: 'Test card' };
    const setHardware = async (overrides) => {
        await testStartup.loginAsUser('admin');
        const { data } = await get('/devices');
        await client.put(`/api/v1/devices/${data.devices[0].id}`, { overrides });
    };

    beforeAll(async () => {
        resetState();
        setOllamaClientFactory(fakeOllama);
        await setHardware(BIG_DEVICE);
    });

    afterAll(async () => {
        state.hold = null;
        state.stall = false;
        state.silent = false;
        await downloads.idle();
        downloads.setDiskProbe(null);
        await setHardware({ memoryMb: null, gpuMemoryMb: null, gpuName: '' });
    });

    beforeEach(async () => {
        state.hold = null;
        state.stall = false;
        state.silent = false;
        await downloads.idle();
        resetState();
        Object.assign(downloads, { stallWarnMs: 20000, stallFailMs: 180000, retryMs: 5000, saveWaitMs: 3000 });
        downloads.setDiskProbe(async () => ({ diskFreeMb: 500000, diskTotalMb: 1000000 }));
        await Download.deleteMany({});
        await PersonalModel.deleteMany({});
    });

    describe('GET /models', () => {
        test('refuses a visitor who is not signed in', async () => {
            signOut();
            expect((await get('/models')).status).toBe(401);
        });

        test('lists installed models with plain details', async () => {
            await testStartup.loginAsUser('user');
            const response = await get('/models');

            expect(response.status).toBe(200);
            expect(response.data.ollamaAvailable).toBe(true);
            expect(response.data.ollamaVersion).toBe('9.9.9-test');
            expect(response.data.permissions.manage).toBe(false);
            expect(response.data.models).toHaveLength(2);

            const known = response.data.models.find((model) => model.name === 'qwen2.5-coder:7b');
            expect(known).toMatchObject({
                displayName: 'Qwen 2.5 Coder 7B',
                family: 'testfam',
                parameterSize: '7.6B',
                quantization: 'Q4_K_M',
                contextLength: 32768,
                personal: false,
                usedBy: []
            });
            expect(known.capabilities).toContain('tools');
            expect(known.sizeBytes).toBe(4700 * MB);

            const unknown = response.data.models.find((model) => model.name === 'custom-thing:latest');
            expect(unknown.displayName).toBe('Custom Thing');
            expect(response.data.disk).toEqual(expect.objectContaining({ usedByModelsMb: 4750 }));
        });

        test('answers with an empty list when Ollama is not running', async () => {
            state.up = false;
            await testStartup.loginAsUser('user');
            const response = await get('/models');
            expect(response.status).toBe(200);
            expect(response.data).toMatchObject({ ollamaAvailable: false, models: [] });
        });
    });

    describe('GET /models/running', () => {
        test('reports how much of each loaded model is on the graphics card', async () => {
            state.running = [
                { name: 'qwen2.5-coder:7b', size: 6000 * MB, size_vram: 6000 * MB },
                { name: 'big:30b', size: 20000 * MB, size_vram: 5000 * MB },
                { name: 'cpuonly:1b', size: 1000 * MB, size_vram: 0 }
            ];
            await testStartup.loginAsUser('user');
            const response = await get('/models/running');

            expect(response.status).toBe(200);
            expect(response.data.models.map((model) => [model.gpuPercent, model.healthy])).toEqual([
                [100, true], [25, false], [0, false]
            ]);
        });

        test('explains plainly when Ollama is not running', async () => {
            state.up = false;
            await testStartup.loginAsUser('user');
            const response = await get('/models/running');
            expect(response.status).toBe(503);
            expect(response.data.code).toBe('OLLAMA_DOWN');
        });
    });

    describe('POST /models/show', () => {
        test('returns the definition of an installed model', async () => {
            await testStartup.loginAsUser('user');
            const response = await post('/models/show', { name: 'qwen2.5-coder:7b' });
            expect(response.status).toBe(200);
            expect(response.data.model).toMatchObject({ definition: 'FROM qwen2.5-coder:7b', contextLength: 32768 });
        });

        test('answers 404 in plain words for a model that is not installed', async () => {
            await testStartup.loginAsUser('user');
            const response = await post('/models/show', { name: 'nope:1b' });
            expect(response.status).toBe(404);
            expect(response.data).toMatchObject({ code: 'MODEL_MISSING' });
        });

        test('rejects a malformed name', async () => {
            await testStartup.loginAsUser('user');
            expect((await post('/models/show', { name: 'bad name!' })).status).toBe(400);
            expect((await post('/models/show', {})).status).toBe(400);
        });
    });

    describe('Changing the device needs manage rights', () => {
        test.each([
            ['/models/delete', { name: 'qwen2.5-coder:7b' }],
            ['/models/unload', { name: 'qwen2.5-coder:7b' }],
            ['/models/pull', { name: 'llama3.2:3b' }],
            ['/models/create', { name: 'mine', displayName: 'Mine', from: 'qwen2.5-coder:7b' }],
            ['/models/downloads/cancel', { id: '64b7f0c2a1b2c3d4e5f60718', action: 'cancel' }]
        ])('a regular user is refused on %s', async (path, body) => {
            await testStartup.loginAsUser('user');
            expect((await post(path, body)).status).toBe(403);
            expect(state.models).toHaveLength(2);
        });

        test.each(['creator', 'superCreator'])('a %s is refused too', async (role) => {
            await testStartup.loginAsUser(role);
            expect((await post('/models/delete', { name: 'qwen2.5-coder:7b' })).status).toBe(403);
        });

        test('a visitor who is not signed in is refused', async () => {
            signOut();
            expect((await post('/models/delete', { name: 'qwen2.5-coder:7b' })).status).toBe(401);
        });
    });

    describe('POST /models/delete and /models/unload', () => {
        test('an admin can delete a model', async () => {
            await testStartup.loginAsUser('admin');
            const response = await post('/models/delete', { name: 'custom-thing:latest' });
            expect(response.status).toBe(200);
            expect(state.models.map((model) => model.name)).toEqual(['qwen2.5-coder:7b']);
        });

        test('a removal or download that names no computer of the person\'s is refused, and nothing is removed', async () => {
            await testStartup.loginAsUser('admin');
            const before = state.models.map((model) => model.name);
            const unnamed = { validateStatus: () => true };
            const elsewhere = { headers: { 'X-Mollama-Device': '64b7f0c2a1b2c3d4e5f60718' }, validateStatus: () => true };
            for (const [path, body] of [['/models/delete', { name: before[0] }], ['/models/pull', { name: 'llama3.2:3b' }]]) {
                for (const config of [unnamed, elsewhere]) {
                    const response = await client.post(`/api/v1${path}`, body, config);
                    expect(response.status).toBe(400);
                    expect(response.data).toMatchObject({ code: 'DEVICE_NOT_NAMED', message: 'Choose the computer first. Nothing was removed or downloaded.' });
                }
            }
            expect(state.models.map((model) => model.name)).toEqual(before);
            expect((await get('/models/downloads')).data.downloads.some((job) => job.model === 'llama3.2:3b' && job.status !== 'cancelled')).toBe(false);
        });

        test('deleting a model that is not there answers 404', async () => {
            await testStartup.loginAsUser('admin');
            const response = await post('/models/delete', { name: 'ghost:1b' });
            expect(response.status).toBe(404);
            expect(response.data.code).toBe('MODEL_MISSING');
        });

        test('an owner can unload a model from memory', async () => {
            await testStartup.loginAsUser('owner');
            const response = await post('/models/unload', { name: 'qwen2.5-coder:7b' });
            expect(response.status).toBe(200);
            expect(state.unloaded).toEqual([{ model: 'qwen2.5-coder:7b', keep_alive: 0 }]);
        });
    });

    describe('POST /models/create', () => {
        const careful = {
            name: 'careful-reviewer',
            displayName: 'Careful reviewer',
            from: 'qwen2.5-coder:7b',
            system: 'Check the work for mistakes.',
            parameters: { temperature: 0.2 }
        };

        test('an admin creates a personal model that shows in the library', async () => {
            await testStartup.loginAsUser('admin');
            const response = await post('/models/create', careful);

            expect(response.status).toBe(201);
            expect(state.created[0]).toMatchObject({
                model: 'careful-reviewer', from: 'qwen2.5-coder:7b', system: 'Check the work for mistakes.',
                parameters: { temperature: 0.2 }, stream: false
            });

            const stored = await PersonalModel.findOne({ name: 'careful-reviewer' });
            expect(stored.service).toBe('ai-base');

            const list = await get('/models');
            const mine = list.data.models.find((model) => model.name === 'careful-reviewer:latest');
            expect(mine).toMatchObject({ personal: true, displayName: 'Careful reviewer', basedOn: 'qwen2.5-coder:7b' });
            const base = list.data.models.find((model) => model.name === 'qwen2.5-coder:7b');
            expect(base.usedBy).toEqual(['Careful reviewer']);
        });

        test('a personal model shows its instructions and can be edited', async () => {
            await testStartup.loginAsUser('admin');
            await post('/models/create', careful);
            const before = (await get('/models')).data.models.find((model) => model.name === 'careful-reviewer:latest');
            expect(before).toMatchObject({ system: 'Check the work for mistakes.', parameters: { temperature: 0.2 } });
            expect((await get('/models')).data.models.find((model) => model.name === 'qwen2.5-coder:7b')).toMatchObject({ system: null, parameters: null });

            const response = await post('/models/update', {
                name: 'careful-reviewer:latest', displayName: 'Very careful reviewer', system: 'Check twice.', parameters: { temperature: 0.7, num_ctx: 4096 }
            });
            expect(response.status).toBe(200);
            expect(state.created[1]).toMatchObject({
                model: 'careful-reviewer', from: 'qwen2.5-coder:7b', system: 'Check twice.', parameters: { temperature: 0.7, num_ctx: 4096 }, stream: false
            });
            const after = (await get('/models')).data.models.find((model) => model.name === 'careful-reviewer:latest');
            expect(after).toMatchObject({ displayName: 'Very careful reviewer', basedOn: 'qwen2.5-coder:7b', system: 'Check twice.', parameters: { temperature: 0.7, num_ctx: 4096 } });

            expect((await post('/models/update', { name: 'qwen2.5-coder:7b', system: 'x' })).status).toBe(404);
            expect((await post('/models/update', { name: 'careful-reviewer', from: 'missing:7b' })).status).toBe(404);
            expect((await post('/models/update', { name: 'careful-reviewer', from: 'careful-reviewer' })).status).toBe(404);
            expect((await post('/models/update', { name: 'careful-reviewer' })).status).toBe(400);
            expect(state.created).toHaveLength(2);

            await testStartup.loginAsUser('user');
            expect((await post('/models/update', { name: 'careful-reviewer', system: 'x' })).status).toBe(403);
        });

        test('refuses a name that is already taken', async () => {
            await testStartup.loginAsUser('admin');
            await post('/models/create', careful);
            const again = await post('/models/create', careful);
            expect(again.status).toBe(409);
        });

        test('refuses a base model that is not installed', async () => {
            await testStartup.loginAsUser('admin');
            const response = await post('/models/create', { ...careful, from: 'missing:7b' });
            expect(response.status).toBe(404);
            expect(response.data.code).toBe('MODEL_MISSING');
            expect(state.created).toHaveLength(0);
        });

        test('deleting a personal model removes its record', async () => {
            await testStartup.loginAsUser('admin');
            await post('/models/create', careful);
            await post('/models/delete', { name: 'careful-reviewer:latest' });
            expect(await PersonalModel.countDocuments({})).toBe(0);
        });

        test('rejects an invalid request', async () => {
            await testStartup.loginAsUser('admin');
            expect((await post('/models/create', { ...careful, name: 'Has Spaces' })).status).toBe(400);
            expect((await post('/models/create', { ...careful, system: 'x'.repeat(20001) })).status).toBe(400);
        });
    });

    describe('Catalogue', () => {
        const everything = async (query = '') => {
            const first = await get(`/catalogue${query ? `?${query}` : ''}`);
            const second = await get(`/catalogue?${query ? `${query}&` : ''}page=2`);
            return { first, second, models: [...first.data.models, ...second.data.models] };
        };

        test('is public and lists the models on ollama.com, a page at a time', async () => {
            signOut();
            const { first, second, models } = await everything();
            expect(first.status).toBe(200);
            expect(first.data).toMatchObject({ source: 'ollama.com', page: 1, more: false, query: '', total: catalogue.length + 1, library: libraryNames().length + 1 });
            expect(second.data).toMatchObject({ page: 2, more: false, models: [] });
            expect(first.data.models.map((model) => model.tag).sort()).toEqual([...catalogue.map((model) => model.tag), 'kimi-k3'].sort());
            expect(first.data.models[first.data.models.length - 1]).toMatchObject({ tag: 'kimi-k3', name: 'Kimi K3', cloudOnly: true, sizeMb: 0 });
            expect((await get('/catalogue?all=true')).data).toMatchObject({ more: false, total: catalogue.length + 1 });
            expect(library.marked).toContain('/search?page=2');
            expect(library.asked).toContain('/search');
            expect(models[0].fit).toBeNull();
            const gemma = models.find((model) => model.tag === 'gemma3:12b');
            expect(gemma).toMatchObject({ name: 'Gemma 3 12B', parameters: 12, sizeEstimated: false, sizeMb: 8100, capabilities: ['vision'], pulls: 1234567, updated: 'October 1, 2026' });
            expect(models.find((model) => model.tag === 'qwen2.5-coder:7b').goodFor).toContain('code');
            expect(models.find((model) => model.tag === 'qwen3:30b').moe).toBe(true);
        });

        test('searches the whole library by name', async () => {
            signOut();
            const found = await get('/catalogue?q=gemma&memoryMb=32768&gpuMemoryMb=8192');
            expect(found.data.query).toBe('gemma');
            expect(found.data.models.map((model) => model.tag).sort()).toEqual(['gemma3:12b', 'gemma3:27b', 'gemma3:4b']);
            expect(found.data.models.every((model) => model.fit.verdict)).toBe(true);
            expect(found.data).toMatchObject({ total: 3, fitting: 3, library: libraryNames().length + 1 });
            const kimi = (await get('/catalogue?q=kimi&memoryMb=32768&gpuMemoryMb=8192')).data;
            expect(kimi.models).toHaveLength(1);
            expect(kimi.models[0]).toMatchObject({ cloudOnly: true, pick: null, fit: { verdict: 'cloud', fits: false, label: 'Cloud only' } });
            expect(kimi.fitting).toBe(0);
            expect((await get('/catalogue?q=coder%20qwen3')).data.models.map((model) => model.tag)).toEqual(['qwen3-coder:30b']);
            expect((await get('/catalogue?q=nothing-like-this')).data.models).toEqual([]);
            // A model too large for the computer described says how many like it would hold it, pooled.
            const tiny = (await get('/catalogue?q=gemma&memoryMb=8192&gpuMemoryMb=0')).data.models;
            expect(tiny.find((model) => model.tag === 'gemma3:12b')).toMatchObject({ fit: { fits: false }, pool: { copies: 3 } });
            expect(tiny.find((model) => model.tag === 'gemma3:4b')).toMatchObject({ fit: { fits: false }, pool: { copies: 2 } });
            expect((await get('/catalogue?page=0')).status).toBe(400);
            expect((await get('/catalogue?size=10')).status).toBe(400);
            const small = await get('/catalogue?size=12');
            expect(small.status).toBe(200);
            expect(small.data.size).toBe(12);
            expect(small.data.models.filter((model) => !model.pick).length).toBeLessThanOrEqual(12);
            expect(small.data.models.filter((model) => model.pick).length).toBeLessThanOrEqual(4);
            expect(small.data.paged).toBe(small.data.total - small.data.models.filter((model) => model.pick).length);
            expect((await get('/catalogue')).data.size).toBe(48);
        });

        test('a search forgives spacing and spelling, reads a size, and puts the closest name first', async () => {
            signOut();
            const tags = async (query) => (await get(`/catalogue?${query}`)).data.models.map((model) => model.tag);
            // The name asked for comes before a longer name that starts the same way, and 30b keeps to that size.
            expect(await tags('q=qwen3%2030b')).toEqual(['qwen3:30b', 'qwen3-coder:30b']);
            expect(await tags('q=qwen3:30b')).toEqual(['qwen3:30b', 'qwen3-coder:30b']);
            // Spaces, dots and hyphens in a name do not have to be typed as the library writes them.
            expect((await tags('q=deep%20seek')).sort()).toEqual(['deepseek-r1:14b', 'deepseek-r1:8b']);
            expect(await tags('q=llama%203.2')).toEqual(['llama3.2:3b']);
            expect((await tags('q=qwen%202.5%20coder')).sort()).toEqual(['qwen2.5-coder:32b', 'qwen2.5-coder:7b']);
            // An ability can be searched for by its name.
            expect((await tags('q=vision')).sort()).toEqual(['gemma3:12b', 'gemma3:27b', 'gemma3:4b']);
            // A slip of a letter still finds the model, and the answer says the match is not exact.
            const slip = (await get('/catalogue?q=gemna')).data;
            expect(slip.near).toBe(true);
            expect(slip.models.map((model) => model.tag).sort()).toEqual(['gemma3:12b', 'gemma3:27b', 'gemma3:4b']);
            expect((await get('/catalogue?q=gemma')).data.near).toBe(false);
            expect((await get('/catalogue?q=zzzzzz')).data).toMatchObject({ near: false, models: [], total: 0 });
        });

        test('the list can be narrowed by what a model is for, what it can do and whether it fits, across every page', async () => {
            signOut();
            const ask = async (query) => (await get(`/catalogue?${query}`)).data;
            const code = await ask('purpose=code&size=4');
            expect(code.total).toBeGreaterThan(0);
            expect(code.total).toBeLessThan(catalogue.length);
            expect(code.models.every((model) => model.goodFor.includes('code'))).toBe(true);
            const seeing = await ask('can=vision');
            expect(seeing.models.map((model) => model.tag).sort()).toEqual(['gemma3:12b', 'gemma3:27b', 'gemma3:4b']);
            expect((await ask('can=tools,thinking')).models.every((model) => model.capabilities.includes('tools') && model.capabilities.includes('thinking'))).toBe(true);
            expect((await ask('can=vision&q=27b')).models.map((model) => model.tag)).toEqual(['gemma3:27b']);
            const small = await ask('fits=true&memoryMb=8192&gpuMemoryMb=4096');
            const whole = await ask('memoryMb=8192&gpuMemoryMb=4096');
            expect(small.total).toBe(whole.fitting);
            expect(small.total).toBeLessThan(whole.total);
            expect(small.models.every((model) => model.fit.fits)).toBe(true);
            // The count is of the whole narrowed list, not of the page in hand.
            const paged = await ask('purpose=chat&size=4');
            expect(paged.models.filter((model) => !model.pick).length).toBeLessThanOrEqual(4);
            expect(paged.paged).toBeGreaterThan(4);
            // Ranges: the download's size, the memory needed to run, and the size in billions of parameters.
            const mid = await ask('minSizeMb=4000&maxSizeMb=10000');
            expect(mid.total).toBeGreaterThan(0);
            expect(mid.models.every((model) => model.sizeMb >= 4000 && model.sizeMb <= 10000)).toBe(true);
            expect(mid.models.some((model) => model.cloudOnly)).toBe(false);
            const big = await ask('minParams=30');
            expect(big.models.map((model) => model.tag).sort()).toEqual(['llama3.3:70b', 'qwen2.5-coder:32b', 'qwen3-coder:30b', 'qwen3:30b']);
            expect((await ask('maxParams=4')).models.every((model) => model.parameters <= 4)).toBe(true);
            const light = await ask('maxMemoryMb=6000&memoryMb=32768&gpuMemoryMb=8192');
            expect(light.total).toBeGreaterThan(0);
            expect(light.models.every((model) => model.fit.memoryNeededMb <= 6000)).toBe(true);
            expect((await ask('maxMemoryMb=6000')).total).toBe(light.total);
            expect((await ask('minParams=7&maxParams=9&can=tools')).models.every((model) => model.parameters >= 7 && model.parameters <= 9 && model.capabilities.includes('tools'))).toBe(true);
            expect((await get('/catalogue?minParams=big')).status).toBe(400);
            expect((await get('/catalogue?can=flying')).status).toBe(400);
            expect((await get('/catalogue?purpose=cooking')).status).toBe(400);
        });

        test('says plainly when ollama.com cannot be reached or its page cannot be read', async () => {
            signOut();
            setLibraryFetch(fakeLibrary);
            library.down = true;
            try {
                const response = await get('/catalogue');
                expect(response.status).toBe(502);
                expect(response.data).toMatchObject({ code: 'LIBRARY_UNREACHABLE' });
                expect(response.data.message).toMatch(/ollama\.com/);
                library.down = false;
                library.empty = true;
                expect((await get('/catalogue')).data).toMatchObject({ code: 'LIBRARY_UNREADABLE' });
            } finally {
                library.down = false;
                library.empty = false;
                setLibraryFetch(fakeLibrary);
            }
            expect((await get('/catalogue')).status).toBe(200);
        });

        test('reads sizes, abilities and counts from the library pages', () => {
            expect(parseParameters('12b')).toBe(12);
            expect(parseParameters('270m')).toBe(0.27);
            expect(parseParameters('e4b')).toBe(4);
            expect(parseParameters('8x7b')).toBe(56);
            expect(parseParameters('29.2M')).toBeNull();
            expect(parseParameters('latest')).toBeNull();
            expect(['gemma3:12b', 'qwen2.5-coder:7b', 'deepseek-r1:14b', 'gpt-oss:20b', 'phi4-mini:3.8b', 'mixtral:8x7b', 'glm-5.3:latest', 'someone/unlisted:q4'].map(friendlyName))
                .toEqual(['Gemma 3 12B', 'Qwen 2.5 Coder 7B', 'Deepseek R1 14B', 'GPT OSS 20B', 'Phi 4 Mini 3.8B', 'Mixtral 8x7B', 'GLM 5.3', 'someone/unlisted:q4']);
            const [model, other] = parseLibrary('<a href="/library/duck" class="x"><h2><span>duck</span></h2><span>tools</span><span>1b</span><span>7b</span><p>Quacks &amp; codes.</p><span title="2,500 downloads"><span>2.5K</span></span></a><a href="/library/duck:7b">tag</a><a href="/library/goose"><p>Honks.</p><span>cloud</span></a>');
            expect(model).toEqual({ name: 'duck', description: 'Quacks & codes.', capabilities: ['tools'], sizes: ['1b', '7b'], pulls: 2500, updated: null });
            expect(other).toMatchObject({ name: 'goose', capabilities: ['cloud'], sizes: [] });
            expect(parseLibrary('<html>nothing</html>')).toEqual([]);
            expect([...parseLibrarySizes('<a href="/library/duck:7b"><span>duck:7b</span> abc • 4.4GB • 8K</a><a href="/library/duck:1b">x • 815MB •</a>')]).toEqual([['7b', 4506], ['1b', 815]]);
        });

        test('gives verdicts for the specs a visitor enters', async () => {
            signOut();
            const { first: response, models } = await everything('memoryMb=32768&gpuMemoryMb=8192');
            expect(response.status).toBe(200);
            const verdict = (tag) => models.find((model) => model.tag === tag).fit.verdict;
            expect(verdict('llama3.1:8b')).toBe('gpu');
            expect(verdict('qwen3-coder:30b')).toBe('mixed');
            expect(verdict('llama3.3:70b')).toBe('too_large');
            expect(response.data.models.slice(0, 3).map((model) => model.pick)).toEqual(['fast', 'balanced', 'best']);
            expect(response.data.reserved).toEqual({ gpuMemoryMb: 1536, memoryMb: 4096 });
            const notes = (await get('/catalogue?memoryMb=32768&gpuMemoryMb=8192&page=2')).data.notes.join(' ');
            expect(notes).toMatch(/mixture-of-experts/);
            expect(response.data.notes.join(' ')).not.toMatch(/two 7 to 8B/);
        });

        test('explains what the specs mean: two models at once, no graphics card, nothing fits', async () => {
            signOut();
            const notes = async (query) => (await get(`/catalogue?${query}`)).data.notes.join(' ');
            expect(await notes('memoryMb=65536&gpuMemoryMb=24576')).toMatch(/two 7 to 8B models at once/);
            const noCard = await get('/catalogue?memoryMb=16384&gpuMemoryMb=0');
            expect(noCard.data.notes.join(' ')).toMatch(/No graphics card/);
            expect(noCard.data.models.filter((model) => model.fit.fits).every((model) => model.fit.verdict === 'cpu')).toBe(true);
            const small = await get('/catalogue?memoryMb=8192&gpuMemoryMb=4096');
            expect(small.data.models.some((model) => model.pick)).toBe(true);
            expect(await notes('memoryMb=1024&gpuMemoryMb=0')).toMatch(/None of the listed models fits/);
            expect((await get('/catalogue')).data.notes).toEqual([]);
        });

        test('gives the home page a sample: models that run fully, some that need system memory, and a few too large', async () => {
            signOut();
            const response = await get('/catalogue?memoryMb=32768&gpuMemoryMb=8192&sample=true');
            expect(response.status).toBe(200);
            const verdicts = response.data.models.map((model) => model.fit.verdict);
            const count = (verdict) => verdicts.filter((found) => found === verdict).length;
            expect(count('gpu')).toBeGreaterThan(0);
            expect(count('gpu')).toBeLessThanOrEqual(6);
            expect(count('mixed')).toBeGreaterThan(0);
            expect(count('mixed')).toBeLessThanOrEqual(3);
            expect(count('too_large')).toBeGreaterThan(0);
            expect(count('too_large')).toBeLessThanOrEqual(3);
            expect(new Set(response.data.models.map((model) => model.tag)).size).toBe(response.data.models.length);
            expect(response.data.models.some((model) => model.cloudOnly)).toBe(false);
            expect(response.data).toMatchObject({ more: false, total: catalogue.length + 1 });
            expect(response.data.fitting).toBeGreaterThan(count('gpu') + count('mixed') - 1);
            expect((await get('/catalogue?sample=true')).data.models.length).toBe(catalogue.length + 1);
        });

        test('a chip whose graphics share the system memory does not have that memory counted twice', async () => {
            const model = { sizeMb: 17000, parameters: 27 };
            const separate = { memoryMb: 24576, gpuMemoryMb: 16384 };
            expect(fitModel(model, separate).verdict).toBe('mixed');
            expect(fitModel(model, { ...separate, gpuShared: true }).verdict).toBe('too_large');
            expect(fitModel({ sizeMb: 8100, parameters: 12 }, { ...separate, gpuShared: true }).verdict).toBe('gpu');

            signOut();
            const apart = (await get('/catalogue?memoryMb=24576&gpuMemoryMb=16384&all=true')).data;
            const shared = (await get('/catalogue?memoryMb=24576&gpuMemoryMb=16384&shared=true&all=true')).data;
            expect(shared.hardware.gpuShared).toBe(true);
            expect(shared.fitting).toBeLessThan(apart.fitting);
            expect(shared.models.find((entry) => entry.tag === 'gemma3:27b').fit.verdict).toBe('too_large');
            expect(apart.models.find((entry) => entry.tag === 'gemma3:27b').fit.verdict).toBe('mixed');
        });

        test('rejects nonsense specs', async () => {
            signOut();
            expect((await get('/catalogue?memoryMb=-5')).status).toBe(400);
            expect((await get('/catalogue?memoryMb=abc')).status).toBe(400);
            expect((await get('/catalogue?somethingElse=1')).status).toBe(400);
        });

        test('the device catalogue needs sign-in and agrees with the public verdicts', async () => {
            signOut();
            const devicesAnon = await get('/devices/64b7f0c2a1b2c3d4e5f60718/catalogue');
            expect(devicesAnon.status).toBe(401);

            await testStartup.loginAsUser('admin');
            const { data: deviceList } = await get('/devices');
            const device = deviceList.devices[0];
            await client.put(`/api/v1/devices/${device.id}`, { overrides: { memoryMb: 32768, gpuMemoryMb: 8192, gpuName: 'Test' } });

            const mine = await get(`/devices/${device.id}/catalogue`);
            expect(mine.status).toBe(200);
            expect(mine.data.permissions.manage).toBe(true);
            expect(mine.data.models.find((model) => model.tag === 'qwen2.5-coder:7b').installed).toBe(true);
            expect(mine.data.models.find((model) => model.tag === 'llama3.2:3b').installed).toBe(false);

            signOut();
            const visitor = await get('/catalogue?memoryMb=32768&gpuMemoryMb=8192');
            const pairs = (data) => data.models.map((model) => [model.tag, model.fit.verdict, model.pick]);
            expect(pairs(mine.data)).toEqual(pairs(visitor.data));

            await setHardware(BIG_DEVICE);
        });

        test('marks models that will not fit on the disk', async () => {
            downloads.setDiskProbe(async () => ({ diskFreeMb: 500000, diskTotalMb: 1000000 }));
            await testStartup.loginAsUser('user');
            const { data: deviceList } = await get('/devices');
            const response = await get(`/devices/${deviceList.devices[0].id}/catalogue`);
            expect(response.data.models.every((model) => typeof model.fitsDisk === 'boolean')).toBe(true);
        });
    });

    describe('Downloads', () => {
        const finished = async (id, status = 'completed') => {
            for (let attempt = 0; attempt < 200; attempt++) {
                const job = (await get('/models/downloads')).data.downloads.find((download) => download.id === id);
                if (job?.status === status) return job;
                await new Promise((resolve) => setTimeout(resolve, 25));
            }
            throw new Error(`download ${id} never became ${status}`);
        };

        const halfway = async (id) => {
            for (let attempt = 0; attempt < 200; attempt++) {
                const job = (await get('/models/downloads')).data.downloads.find((download) => download.id === id);
                if (job?.status === 'downloading' && job.percent >= 50) return job;
                await new Promise((resolve) => setTimeout(resolve, 25));
            }
            throw new Error(`download ${id} never reached halfway`);
        };

        test('a finished download and a finished reply are announced over the notification socket', async () => {
            await testStartup.loginAsUser('admin');
            const token = (await client.get('/api/v1/auth/ws-token')).data.token;
            const messages = [];
            const ws = new WebSocket(`ws://localhost:${testStartup.port}/notifications?token=${encodeURIComponent(token)}`);
            ws.addEventListener('message', (event) => messages.push(JSON.parse(event.data)));
            await new Promise((resolve, reject) => {
                ws.addEventListener('open', resolve);
                ws.addEventListener('error', () => reject(new Error('socket error')));
            });
            const seen = async (type) => {
                for (let attempt = 0; attempt < 200; attempt++) {
                    const found = messages.find((message) => message.type === type);
                    if (found) return found;
                    await new Promise((resolve) => setTimeout(resolve, 25));
                }
                throw new Error(`no ${type}; saw ${messages.map((message) => message.type).join(', ')}`);
            };

            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await seen('activity:changed');
            const finished = await seen('download:finished');
            expect(finished.data).toMatchObject({ level: 'success', message: 'llama3.2:3b has finished downloading.', model: 'llama3.2:3b', id: job.id });
            expect(messages.filter((message) => message.type === 'download:finished')).toHaveLength(1);

            const bad = (await post('/models/pull', { name: 'nosuch-model:9b' })).data.download;
            const failed = await seen('download:failed');
            expect(failed.data).toMatchObject({ level: 'error', id: bad.id });
            expect(failed.data.message).toMatch(/nosuch-model:9b could not be downloaded/);

            const chat = (await post('/chats', { model: 'qwen2.5-coder:7b' })).data.chat;
            const answer = await client.post(`/api/v1/chats/${chat.id}/messages`, { content: 'hello there' });
            expect(answer.status).toBe(200);
            const ready = await seen('reply:finished');
            expect(ready.data).toMatchObject({ level: 'info', chatId: chat.id, path: `/chat/${chat.id}` });
            expect(ready.data.message).toContain('hello there');
            ws.close();
        });

        test('an admin starts a download that finishes in the background', async () => {
            await testStartup.loginAsUser('admin');
            const response = await post('/models/pull', { name: 'llama3.2:3b' });

            expect(response.status).toBe(202);
            expect(response.data.download).toMatchObject({ model: 'llama3.2:3b' });
            expect(['queued', 'downloading']).toContain(response.data.download.status);

            const job = await finished(response.data.download.id);
            expect(job).toMatchObject({ percent: 100, completedBytes: 10 * MB, totalBytes: 10 * MB, error: null });
            expect(state.models.some((model) => model.name === 'llama3.2:3b')).toBe(true);

            const stored = await Download.findById(job.id);
            expect(stored.service).toBe('ai-base');
            expect(stored.finishedAt).toBeInstanceOf(Date);
        });

        test('refuses a download that will not fit, before it starts', async () => {
            downloads.setDiskProbe(async () => ({ diskFreeMb: 1500, diskTotalMb: 100000 }));
            await testStartup.loginAsUser('admin');
            const response = await post('/models/pull', { name: 'llama3.3:70b' });

            expect(response.status).toBe(422);
            expect(response.data).toMatchObject({ code: 'DISK_FULL', neededMb: 43000, freeMb: 1500 });
            expect(response.data.message).toMatch(/42 GB|43 GB/);
            expect(await Download.countDocuments({})).toBe(0);
        });

        test('stops an unlisted model that turns out to be too large', async () => {
            downloads.setDiskProbe(async () => ({ diskFreeMb: 2, diskTotalMb: 100000 }));
            state.pullSizeMb = 50;
            await testStartup.loginAsUser('admin');
            const response = await post('/models/pull', { name: 'someone/unlisted:q4' });
            const job = await finished(response.data.download.id, 'failed');
            expect(job.error.code).toBe('DISK_FULL');
            expect(state.models.some((model) => model.name === 'someone/unlisted:q4')).toBe(false);
        });

        test('runs one download at a time and queues the next', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const first = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(first.id);
            const second = (await post('/models/pull', { name: 'mistral:7b' })).data.download;

            const during = (await get('/models/downloads')).data.downloads;
            expect(during.find((job) => job.id === first.id)).toMatchObject({ status: 'downloading', percent: 50 });
            expect(during.find((job) => job.id === second.id).status).toBe('queued');

            state.hold = null;
            await finished(first.id);
            await finished(second.id);
            expect(state.models.map((model) => model.name)).toEqual(expect.arrayContaining(['llama3.2:3b', 'mistral:7b']));
        });

        test('asking again for the same model does not start a second download', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const first = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            const again = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            expect(again.id).toBe(first.id);
            expect(await Download.countDocuments({})).toBe(1);
            state.hold = null;
            await finished(first.id);
        });

        test('pauses a download and resumes it', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(job.id);

            const paused = await post('/models/downloads/cancel', { id: job.id, action: 'pause' });
            expect(paused.status).toBe(200);
            expect(paused.data.download).toMatchObject({ status: 'paused', percent: 50 });
            expect(state.models.some((model) => model.name === 'llama3.2:3b')).toBe(false);

            state.hold = null;
            const resumed = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            expect(resumed.id).toBe(job.id);
            await finished(job.id);
        });

        test('cancels a download', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(job.id);

            const cancelled = await post('/models/downloads/cancel', { id: job.id, action: 'cancel' });
            expect(cancelled.data.download.status).toBe('cancelled');
            state.hold = null;
            await downloads.idle();
            expect(state.models.some((model) => model.name === 'llama3.2:3b')).toBe(false);
        });

        test('cancels a download that is still waiting in the queue', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const first = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(first.id);
            const second = (await post('/models/pull', { name: 'mistral:7b' })).data.download;

            const cancelled = await post('/models/downloads/cancel', { id: second.id, action: 'cancel' });
            expect(cancelled.data.download.status).toBe('cancelled');
            state.hold = null;
            await finished(first.id);
            await downloads.idle();
            expect(state.models.some((model) => model.name === 'mistral:7b')).toBe(false);
        });

        test('reports an interrupted download in plain words and lets it resume', async () => {
            state.pullFailsMidway = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            const failed = await finished(job.id, 'failed');
            expect(failed.error).toMatchObject({ code: 'DOWNLOAD_INTERRUPTED' });
            expect(failed.error.action).toMatch(/resume/i);

            state.pullFailsMidway = false;
            const retry = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await finished(retry.id);
        });

        test('reports that Ollama is not running', async () => {
            state.up = false;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            const failed = await finished(job.id, 'failed');
            expect(failed.error.code).toBe('OLLAMA_DOWN');
        });

        test('picks up a download that was interrupted by a restart', async () => {
            await testStartup.loginAsUser('admin');
            const stale = await Download.create({
                model: 'llama3.2:3b', status: 'downloading', completedBytes: 3 * MB, totalBytes: 10 * MB,
                requestedBy: testStartup.admin.id || testStartup.admin._id
            });
            await downloads.recover();
            const job = await finished(stale.id);
            expect(job.percent).toBe(100);
        });

        test('a regular user can watch downloads but not change them', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(job.id);

            await testStartup.loginAsUser('user');
            const list = await get('/models/downloads');
            expect(list.status).toBe(200);
            expect(list.data.permissions.manage).toBe(false);
            expect(list.data.downloads[0].id).toBe(job.id);
            expect((await post('/models/downloads/cancel', { id: job.id, action: 'cancel' })).status).toBe(403);

            state.hold = null;
            await testStartup.loginAsUser('admin');
            await finished(job.id);
        });

        test('rejects a malformed cancel request and an unknown download', async () => {
            await testStartup.loginAsUser('admin');
            expect((await post('/models/downloads/cancel', { id: 'nope', action: 'cancel' })).status).toBe(400);
            expect((await post('/models/downloads/cancel', { id: '64b7f0c2a1b2c3d4e5f60718', action: 'explode' })).status).toBe(400);
            expect((await post('/models/downloads/cancel', { id: '64b7f0c2a1b2c3d4e5f60718', action: 'cancel' })).status).toBe(404);
        });

        const until = async (id, check, what) => {
            for (let attempt = 0; attempt < 200; attempt++) {
                const job = (await get('/models/downloads')).data.downloads.find((download) => download.id === id);
                if (job && check(job)) return job;
                await new Promise((resolve) => setTimeout(resolve, 25));
            }
            throw new Error(`download ${id} never ${what}`);
        };

        test('says so when no data is arriving, then carries on when it returns', async () => {
            downloads.stallWarnMs = 100;
            state.stall = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;

            const stalled = await until(job.id, (current) => current.stalled && current.percent >= 50, 'reported a stall');
            expect(stalled).toMatchObject({ status: 'downloading', percent: 50, speedBps: 0, error: null });

            state.stall = false;
            const done = await finished(job.id);
            expect(done.stalled).toBe(false);
        });

        test('a stalled download can still be paused and cancelled', async () => {
            downloads.stallWarnMs = 100;
            state.stall = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await until(job.id, (current) => current.stalled, 'reported a stall');

            const paused = await post('/models/downloads/cancel', { id: job.id, action: 'pause' });
            expect(paused.data.download).toMatchObject({ status: 'paused', stalled: false });

            const cancelled = await post('/models/downloads/cancel', { id: job.id, action: 'cancel' });
            expect(cancelled.data.download.status).toBe('cancelled');
        });

        test('gives up on a download that stays stalled, in plain words, and lets it resume', async () => {
            downloads.stallWarnMs = 100;
            downloads.stallFailMs = 400;
            state.stall = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;

            const failed = await finished(job.id, 'failed');
            expect(failed.error.code).toBe('DOWNLOAD_INTERRUPTED');
            expect(failed.error.action).toMatch(/resume/i);
            expect(failed.stalled).toBe(false);

            state.stall = false;
            const retry = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            expect(retry.id).toBe(job.id);
            expect(retry.error).toBeNull();
            await finished(retry.id);
            expect(await Download.countDocuments({ model: 'llama3.2:3b' })).toBe(1);
        });

        test('gives up when a download never gets going', async () => {
            downloads.stallWarnMs = 100;
            downloads.stallFailMs = 300;
            state.silent = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            const failed = await finished(job.id, 'failed');
            expect(failed.error.code).toBe('DOWNLOAD_INTERRUPTED');
            state.silent = false;
        });

        test('keeps downloading when progress cannot be saved, and saves once it can', async () => {
            const realUpdate = Download.updateOne;
            downloads.retryMs = 50;
            downloads.saveWaitMs = 200;
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(job.id);

            try {
                Download.updateOne = () => Promise.reject(new Error('database is unreachable'));
                state.hold = null;

                const shown = await finished(job.id);
                expect(shown.percent).toBe(100);
                expect(state.models.some((model) => model.name === 'llama3.2:3b')).toBe(true);
                expect((await Download.findById(job.id)).status).not.toBe('completed');

                const next = (await post('/models/pull', { name: 'mistral:7b' })).data.download;
                await finished(next.id);
            } finally {
                Download.updateOne = realUpdate;
            }

            for (let attempt = 0; attempt < 200; attempt++) {
                if ((await Download.findById(job.id)).status === 'completed') break;
                await new Promise((resolve) => setTimeout(resolve, 25));
            }
            const stored = await Download.findById(job.id);
            expect(stored.status).toBe('completed');
            expect(stored.finishedAt).toBeInstanceOf(Date);
        });

        test('a download whose result could not be saved can still be paused', async () => {
            const realUpdate = Download.updateOne;
            downloads.retryMs = 50;
            downloads.saveWaitMs = 200;
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await halfway(job.id);

            try {
                Download.updateOne = () => Promise.reject(new Error('database is unreachable'));
                const paused = await post('/models/downloads/cancel', { id: job.id, action: 'pause' });
                expect(paused.data.download.status).toBe('paused');
            } finally {
                Download.updateOne = realUpdate;
            }

            state.hold = null;
            const resumed = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            expect(resumed.id).toBe(job.id);
            await finished(job.id);
        });

        test('says plainly when the database cannot be reached', async () => {
            const realCheck = health.databaseReady;
            await testStartup.loginAsUser('admin');
            try {
                health.databaseReady = () => false;
                for (const response of [await get('/models/downloads'), await get('/devices'), await post('/models/pull', { name: 'llama3.2:3b' })]) {
                    expect(response.status).toBe(503);
                    expect(response.data).toMatchObject({ success: false, code: 'DATABASE_DOWN' });
                    expect(response.data.action).toMatch(/internet/i);
                }
            } finally {
                health.databaseReady = realCheck;
            }
            expect((await get('/models/downloads')).status).toBe(200);
        });

        test('refuses a listed model that is too large for the device, before it is queued', async () => {
            await setHardware({ memoryMb: 32768, gpuMemoryMb: 8192, gpuName: 'Test card' });
            const response = await post('/models/pull', { name: 'llama3.3:70b' });
            await setHardware(BIG_DEVICE);

            expect(response.status).toBe(422);
            expect(response.data).toMatchObject({ success: false, code: 'MODEL_TOO_LARGE' });
            expect(response.data.message).toMatch(/8 GB of graphics memory and 32 GB of system memory/);
            expect(response.data.action).toMatch(/smaller/i);
            expect(await Download.countDocuments({})).toBe(0);
        });

        test('still queues a listed model that fits', async () => {
            await setHardware({ memoryMb: 32768, gpuMemoryMb: 8192, gpuName: 'Test card' });
            const response = await post('/models/pull', { name: 'qwen3-coder:30b' });
            await setHardware(BIG_DEVICE);
            expect(response.status).toBe(202);
            await finished(response.data.download.id);
        });

        test('stops an unlisted model as soon as it proves too large for the device', async () => {
            downloads.setDiskProbe(async () => ({ diskFreeMb: 50000000, diskTotalMb: 90000000 }));
            state.pullSizeMb = 4000000;
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'someone/enormous:q8' })).data.download;

            const failed = await finished(job.id, 'failed');
            expect(failed.error).toMatchObject({ code: 'MODEL_TOO_LARGE' });
            expect(failed.error.message).toMatch(/too large for this device/i);
            expect(state.models.some((model) => model.name === 'someone/enormous:q8')).toBe(false);
        });

        test('says plainly when no model has that name', async () => {
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'nosuch-model:9b' })).data.download;

            const failed = await finished(job.id, 'failed');
            expect(failed.error.code).toBe('MODEL_NOT_FOUND');
            expect(failed.error.message).toMatch(/no model with this name/i);
            expect(failed.error.message).not.toMatch(/not installed/i);
            expect(failed.error.action).toMatch(/spelling/i);
        });

        test('a failed download can be removed from the list', async () => {
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'nosuch-model:9b' })).data.download;
            await finished(job.id, 'failed');

            const removed = await post('/models/downloads/cancel', { id: job.id, action: 'cancel' });
            expect(removed.status).toBe(200);
            expect(removed.data.download).toMatchObject({ status: 'cancelled', error: expect.anything() });

            const listed = (await get('/models/downloads')).data.downloads.find((download) => download.id === job.id);
            expect(listed.status).toBe('cancelled');
            expect((await Download.findById(job.id)).status).toBe('cancelled');
        });

        test('a finished download clears older failed rows for the same model', async () => {
            await testStartup.loginAsUser('admin');
            const first = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await finished(first.id);
            expect(await downloads.idle()).toBe(true);
            const old = await Download.create([
                { model: 'llama3.2:3b', status: 'failed', errorCode: 'OLLAMA_DOWN', requestedBy: first.requestedBy },
                { model: 'llama3.2:3b', status: 'failed', errorCode: 'DOWNLOAD_INTERRUPTED', requestedBy: first.requestedBy },
                { model: 'mistral:7b', status: 'failed', errorCode: 'OLLAMA_DOWN', requestedBy: first.requestedBy }
            ]);

            const retry = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await finished(retry.id);

            const left = await Download.find({ model: 'llama3.2:3b', status: 'failed' });
            expect(left).toHaveLength(0);
            expect(old.map((doc) => doc.id)).toContain(retry.id);
            expect((await Download.findById(old[2].id)).status).toBe('failed');
        });

        test('a failed download cannot be paused', async () => {
            await testStartup.loginAsUser('admin');
            const job = (await post('/models/pull', { name: 'nosuch-model:9b' })).data.download;
            await finished(job.id, 'failed');
            const response = await post('/models/downloads/cancel', { id: job.id, action: 'pause' });
            expect(response.data.download.status).toBe('failed');
        });

        test('a download that failed does not block the next one', async () => {
            await testStartup.loginAsUser('admin');
            const bad = (await post('/models/pull', { name: 'nosuch-model:9b' })).data.download;
            const good = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await finished(bad.id, 'failed');
            await finished(good.id);
        });

        test('streams progress as server-sent events', async () => {
            state.hold = true;
            await testStartup.loginAsUser('admin');
            const cookies = client.getCookiesForWebSocket();
            const controller = new AbortController();
            const response = await fetch(`${testStartup.baseURL}/api/v1/models/downloads/stream`, {
                headers: { Cookie: cookies }, signal: controller.signal
            });
            expect(response.status).toBe(200);
            expect(response.headers.get('content-type')).toMatch(/text\/event-stream/);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let text = '';
            const readUntil = async (marker) => {
                for (let attempt = 0; attempt < 100 && !text.includes(marker); attempt++) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    text += decoder.decode(value, { stream: true });
                }
            };

            await readUntil('event: snapshot');
            const job = (await post('/models/pull', { name: 'llama3.2:3b' })).data.download;
            await readUntil('"status":"downloading"');
            expect(text).toContain('event: snapshot');
            expect(text).toContain('event: progress');
            expect(text).toContain(job.id);

            controller.abort();
            state.hold = null;
            await finished(job.id);
        });
    });
});

describe('Generation queue', () => {
    const gate = () => {
        let open;
        const waiting = new Promise((resolve) => {
            open = resolve;
        });
        return { open, waiting };
    };
    const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

    test('runs one job at a time by default and reports the place in line', async () => {
        const line = new GenerationQueue();
        const first = gate();
        const ran = [];
        const a = line.add({ userId: 'u1', run: async () => { ran.push('a'); await first.waiting; return 'A'; } });
        const b = line.add({ userId: 'u2', run: async () => { ran.push('b'); return 'B'; } });
        await settle();

        expect(ran).toEqual(['a']);
        expect(a.ahead()).toBe(0);
        expect(b.ahead()).toBe(1);

        first.open();
        expect(await a.done).toBe('A');
        expect(await b.done).toBe('B');
    });

    test('runs chat ahead of background work', async () => {
        const line = new GenerationQueue();
        const first = gate();
        const ran = [];
        line.add({ userId: 'u1', run: () => first.waiting });
        await settle();
        const background = line.add({ userId: 'u1', priority: 'background', run: async () => ran.push('background') });
        const chat = line.add({ userId: 'u2', priority: 'interactive', run: async () => ran.push('chat') });

        expect(chat.ahead()).toBe(1);
        expect(background.ahead()).toBe(2);
        first.open();
        await Promise.all([background.done, chat.done]);
        expect(ran).toEqual(['chat', 'background']);
    });

    test('alternates between two users', async () => {
        const line = new GenerationQueue();
        const first = gate();
        const ran = [];
        line.add({ userId: 'blocker', run: () => first.waiting });
        await settle();
        const jobs = [
            line.add({ userId: 'a', run: async () => ran.push('a1') }),
            line.add({ userId: 'a', run: async () => ran.push('a2') }),
            line.add({ userId: 'a', run: async () => ran.push('a3') }),
            line.add({ userId: 'b', run: async () => ran.push('b1') }),
            line.add({ userId: 'b', run: async () => ran.push('b2') })
        ];
        first.open();
        await Promise.all(jobs.map((job) => job.done));
        expect(ran).toEqual(['a1', 'b1', 'a2', 'b2', 'a3']);
    });

    test('refuses a sixth waiting job from one user, but not from another', async () => {
        const line = new GenerationQueue();
        const first = gate();
        line.add({ userId: 'a', run: () => first.waiting });
        await settle();
        for (let count = 0; count < 5; count++) line.add({ userId: 'a', run: async () => {} });

        expect(() => line.add({ userId: 'a', run: async () => {} })).toThrow(OllamaError);
        try {
            line.add({ userId: 'a', run: async () => {} });
        } catch (error) {
            expect(error.code).toBe('QUEUE_FULL');
            expect(error.status).toBe(429);
        }
        expect(() => line.add({ userId: 'b', run: async () => {} })).not.toThrow();
        first.open();
    });

    test('runs several at once when allowed', async () => {
        const line = new GenerationQueue({ maxConcurrent: 2 });
        const first = gate();
        const ran = [];
        line.add({ userId: 'a', run: async () => { ran.push('a'); await first.waiting; } });
        line.add({ userId: 'b', run: async () => { ran.push('b'); await first.waiting; } });
        const third = line.add({ userId: 'c', run: async () => ran.push('c') });
        await settle();
        expect(ran).toEqual(['a', 'b']);
        expect(third.ahead()).toBe(2);
        first.open();
        await third.done;
    });

    test('each computer has its own line, so one never waits for another', async () => {
        const line = new GenerationQueue();
        const first = gate();
        const ran = [];
        line.add({ userId: 'a', lane: 'mac', run: async () => { ran.push('mac'); await first.waiting; } });
        const other = line.add({ userId: 'a', lane: 'pc', run: async () => ran.push('pc') });
        const behind = line.add({ userId: 'b', lane: 'mac', run: async () => ran.push('mac again') });
        await other.done;
        expect(ran).toEqual(['mac', 'pc']);
        expect(behind.ahead()).toBe(1);
        expect(line.runningIn('mac')).toBe(1);
        await settle();
        expect(line.busyIn('pc')).toBe(false);
        first.open();
        await behind.done;
        expect(ran).toEqual(['mac', 'pc', 'mac again']);
    });

    test('a cancelled job never runs, and a failed job does not block the next', async () => {
        const line = new GenerationQueue();
        const first = gate();
        const ran = [];
        line.add({ userId: 'a', run: () => first.waiting });
        await settle();
        const dropped = line.add({ userId: 'b', run: async () => ran.push('dropped') });
        const failing = line.add({ userId: 'c', run: async () => { throw new Error('boom'); } });
        const last = line.add({ userId: 'd', run: async () => ran.push('last') });

        expect(dropped.cancel()).toBe(true);
        expect(await dropped.done).toBeNull();
        first.open();
        await expect(failing.done).rejects.toThrow('boom');
        await last.done;
        expect(ran).toEqual(['last']);
    });
});

const serverRequire = createRequire(new URL('../../server/package.json', import.meta.url));

const makePdf = (text) => {
    const stream = `BT /F1 12 Tf 72 720 Td (${text}) Tj ET`;
    const objects = [
        '<< /Type /Catalog /Pages 2 0 R >>',
        '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>',
        `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`,
        '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'
    ];
    let body = '%PDF-1.4\n';
    const offsets = [];
    objects.forEach((object, index) => {
        offsets.push(body.length);
        body += `${index + 1} 0 obj\n${object}\nendobj\n`;
    });
    const xref = body.length;
    body += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
    offsets.forEach((offset) => {
        body += `${String(offset).padStart(10, '0')} 00000 n \n`;
    });
    body += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
    return Buffer.from(body, 'latin1');
};

const makeDocx = async (text) => {
    const JSZip = serverRequire('jszip');
    const zip = new JSZip();
    zip.file('[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>');
    zip.file('_rels/.rels', '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>');
    zip.file('word/document.xml', `<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>${text}</w:t></w:r></w:p></w:body></w:document>`);
    return zip.generateAsync({ type: 'nodebuffer' });
};

describe('Reading text from documents', () => {
    test.each([
        ['notes.txt', 'Plain notes about the harbour survey.'],
        ['notes.md', '# Survey\n\nThe **harbour** is deep.'],
        ['figures.csv', 'port,depth\nHarwich,12\nDover,9']
    ])('reads %s as it is', async (name, content) => {
        expect(await extractText({ buffer: Buffer.from(content), name })).toBe(content);
    });

    test('reads the text of a PDF', async () => {
        const text = await extractText({ buffer: makePdf('The quarterly harbour report'), name: 'report.pdf' });
        expect(text).toContain('The quarterly harbour report');
    });

    test('reads the text of a Word file', async () => {
        const text = await extractText({ buffer: await makeDocx('Minutes of the harbour board'), name: 'minutes.docx' });
        expect(text).toContain('Minutes of the harbour board');
    });

    test('removes a byte order mark and Windows line endings', async () => {
        const text = await extractText({ buffer: Buffer.from('﻿one\r\ntwo'), name: 'a.txt' });
        expect(text).toBe('one\ntwo');
    });

    test.each([
        ['photo.png', Buffer.from([0x89, 0x50, 0x4e, 0x47, 0, 0, 0, 13, 0, 1, 2, 3])],
        ['clip.mp4', Buffer.from([0, 0, 0, 24, 0x66, 0x74, 0x79, 0x70, 0, 0, 0, 0])],
        ['old.doc', Buffer.from('looks like text but is an old Word file')],
        ['broken.pdf', Buffer.from('this is not a pdf at all')],
        ['empty.txt', Buffer.from('   \n  ')]
    ])('refuses %s in plain words', async (name, buffer) => {
        await expect(extractText({ buffer, name })).rejects.toMatchObject({ code: 'FILE_UNREADABLE', status: 415 });
    });

    test('refuses a file over 20 MB', async () => {
        const buffer = Buffer.alloc(MAX_ATTACHMENT_BYTES + 1, 'a');
        await expect(extractText({ buffer, name: 'huge.txt' })).rejects.toMatchObject({ code: 'FILE_TOO_LARGE', status: 413 });
    });

    test('estimates tokens from length', () => {
        expect(estimateTokens('')).toBe(0);
        expect(estimateTokens('abcd')).toBe(1);
        expect(estimateTokens('a'.repeat(4001))).toBe(1001);
    });
});

describe('Chat validation', () => {
    const ok = (schema, value) => expect(schema.validate(value).error).toBeUndefined();
    const bad = (schema, value) => expect(schema.validate(value).error).toBeDefined();

    test('a message is 1 to 100,000 characters', () => {
        ok(chatSchemas.send, { content: 'hello' });
        ok(chatSchemas.send, { content: 'a'.repeat(100000) });
        bad(chatSchemas.send, { content: '' });
        bad(chatSchemas.send, { content: 'a'.repeat(100001) });
        bad(chatSchemas.send, {});
    });

    test('regenerate and edit are checked', () => {
        ok(chatSchemas.send, { regenerate: true });
        ok(chatSchemas.send, { content: 'again', editIndex: 0 });
        bad(chatSchemas.send, { regenerate: true, content: 'x' });
        bad(chatSchemas.send, { content: 'x', editIndex: -1 });
        bad(chatSchemas.send, { content: 'x', editIndex: 200 });
        bad(chatSchemas.send, { content: 'x', role: 'system' });
    });

    test('generation options stay inside their ranges', () => {
        ok(chatSchemas.create, { options: { temperature: 0, num_ctx: 512 } });
        ok(chatSchemas.create, { options: { temperature: 2, num_ctx: 262144, top_p: 1, top_k: 500, repeat_penalty: 5, num_predict: -1 } });
        for (const options of [{ temperature: 2.1 }, { temperature: -1 }, { num_ctx: 511 }, { num_ctx: 262145 }, { top_p: 1.1 }, { top_k: 501 }, { num_predict: 100001 }, { made_up: 1 }]) {
            bad(chatSchemas.create, { options });
        }
    });

    test('titles, instructions and file paths are limited', () => {
        ok(chatSchemas.update, { title: 'Plan' });
        bad(chatSchemas.update, {});
        bad(chatSchemas.update, { title: 'a'.repeat(121) });
        bad(chatSchemas.update, { system: 'a'.repeat(100001) });
        ok(chatSchemas.attachFile, { path: '/docs/a.md' });
        for (const path of ['docs/a.md', '/docs//a.md', '', `/${'a'.repeat(1000)}`]) bad(chatSchemas.attachFile, { path });
        bad(chatSchemas.fileSave, { folder: '/docs', name: 'a/b.md', content: 'x' });
        ok(chatSchemas.fileSave, { folder: '/docs', name: 'chat.md', content: 'x' });
    });
});

describe('Chat endpoints', () => {
    const post = (path, body, config) => client.post(`/api/v1${path}`, body, config);
    const get = (path) => client.get(`/api/v1${path}`);
    const patch = (path, body) => client.patch(`/api/v1${path}`, body);
    const remove = (path) => client.delete(`/api/v1${path}`);
    const events = (response) => String(response.data).trim().split('\n').map((line) => JSON.parse(line));
    const newChat = async (body = {}) => (await post('/chats', body)).data.chat;
    const send = async (chatId, body) => {
        const response = await post(`/chats/${chatId}/messages`, body, { responseType: 'text', transformResponse: [(data) => data] });
        return { response, events: response.status === 200 ? events(response) : [] };
    };
    const upload = (chatId, name, buffer) => post(`/chats/${chatId}/attachments?name=${encodeURIComponent(name)}`, buffer, {
        headers: { 'Content-Type': 'application/octet-stream' }
    });
    const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    let filesCalls = [];
    const fsFiles = {
        '/docs/brief.md': { type: 'text', content: 'The client wants a harbour survey by June.' },
        '/docs/report.pdf': { type: 'binary', bytes: makePdf('Dredging costs rose in March') },
        '/docs/photo.png': { type: 'binary', bytes: Buffer.from([0x89, 0x50, 0x4e, 0x47, 0, 0, 0, 13, 0, 1, 2, 3]) }
    };
    const json = (status, body) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
    const fakeFilesystemOne = async (url, options) => {
        const cookie = options.headers.Cookie || '';
        filesCalls.push({ url, method: options.method || 'GET', cookie, csrf: options.headers['X-CSRF-Token'], body: options.body });
        if (state.filesDown) throw new TypeError('fetch failed');
        // A FilesystemOne still waking up answers 502 for a while.
        if (state.filesWaking > 0) {
            state.filesWaking -= 1;
            return json(502, { message: 'Bad Gateway' });
        }
        const address = new URL(url);
        if (address.pathname.endsWith('/directory/contents')) {
            const folder = address.searchParams.get('filePath');
            if ([...address.searchParams.keys()].join() !== 'filePath') return json(400, { success: false, message: 'Validation error' });
            if (folder === '/private') return json(404, { success: false, message: 'Directory not found or access denied' });
            return json(200, { success: true, contents: [
                { fileName: 'brief.md', filePath: '/docs/brief.md', type: 'text', size: 42 },
                { fileName: 'docs', filePath: '/docs', type: 'directory', size: 0 }
            ] });
        }
        if ((options.method || 'GET') === 'POST') {
            const body = JSON.parse(options.body);
            if (body.filePath.startsWith('/locked')) return json(403, { success: false });
            return json(201, { success: true, file: { id: 'f1', filePath: body.filePath, type: 'text' } });
        }
        const [, encoded, part] = address.pathname.match(/\/api\/v1\/files\/([^/]+)\/(content|download)$/) || [];
        const file = fsFiles[decodeURIComponent(encoded || '')];
        if (!file) return json(404, { success: false, message: 'File not found or access denied' });
        if (part === 'content') return json(200, { success: true, fileType: file.type, content: file.type === 'text' ? file.content : { type: 'Buffer', data: [] } });
        return new Response(file.bytes, { status: 200, headers: { 'Content-Length': String(file.bytes.length) } });
    };

    beforeAll(async () => {
        resetState();
        setOllamaClientFactory(fakeOllama);
        setFilesFetch(fakeFilesystemOne);
    });

    afterAll(async () => {
        state.chatHold = false;
        setFilesFetch(null);
        queue.maxConcurrent = 1;
    });

    beforeEach(async () => {
        state.chatHold = false;
        resetState();
        filesCalls = [];
        await Chat.deleteMany({});
        await Prompt.deleteMany({});
        await Setting.deleteMany({});
        await testStartup.loginAsUser('user');
    });

    describe('Creating and listing', () => {
        test('needs sign-in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            expect((await get('/chats')).status).toBe(401);
            expect((await post('/chats', {})).status).toBeGreaterThanOrEqual(401);
        });

        test('a chat can be created with nothing chosen', async () => {
            const response = await post('/chats', {});
            expect(response.status).toBe(201);
            expect(response.data.chat).toMatchObject({ title: 'New chat', model: null, system: '', messages: [], pendingAttachments: [] });
            const stored = await Chat.findById(response.data.chat.id);
            expect(stored.service).toBe('ai-base');
        });

        test('lists only the caller’s chats, newest first, and never caches', async () => {
            const mine = await newChat({ title: 'Mine' });
            await testStartup.loginAsUser('owner');
            await newChat({ title: 'Owner only' });
            await testStartup.loginAsUser('user');
            const second = await newChat({ title: 'Second' });

            const response = await get('/chats');
            expect(response.headers['cache-control']).toMatch(/no-store/);
            expect(response.data.chats.map((chat) => chat.id)).toEqual([second.id, mine.id]);
        });

        test('searches by title', async () => {
            await newChat({ title: 'Harbour survey' });
            await newChat({ title: 'Budget (2026)' });
            expect((await get('/chats?search=harb')).data.chats.map((chat) => chat.title)).toEqual(['Harbour survey']);
            expect((await get('/chats?search=(2026)')).data.chats).toHaveLength(1);
            expect((await get('/chats?search=zzz')).data.chats).toEqual([]);
        });

        test('renames, changes instructions and deletes', async () => {
            const chat = await newChat();
            const renamed = await patch(`/chats/${chat.id}`, { title: 'Renamed', system: 'Reply formally.', options: { temperature: 0.2 } });
            expect(renamed.data.chat).toMatchObject({ title: 'Renamed', system: 'Reply formally.', options: { temperature: 0.2 } });
            expect((await remove(`/chats/${chat.id}`)).status).toBe(200);
            expect((await get(`/chats/${chat.id}`)).status).toBe(404);
        });

        test('rejects a malformed id before it reaches the database', async () => {
            expect((await get('/chats/not-an-id')).status).toBe(400);
        });
    });

    describe('Ownership', () => {
        test('another user cannot read, change, send to, attach to, export or delete a chat', async () => {
            await testStartup.loginAsUser('owner');
            const chat = await newChat({ title: 'Private', model: 'qwen2.5-coder:7b' });
            await send(chat.id, { content: 'secret plan' });

            for (const role of ['user', 'admin']) {
                await testStartup.loginAsUser(role);
                expect((await get(`/chats/${chat.id}`)).status).toBe(404);
                expect((await patch(`/chats/${chat.id}`, { title: 'Taken' })).status).toBe(404);
                expect((await send(chat.id, { content: 'hello' })).response.status).toBe(404);
                expect((await send(chat.id, { regenerate: true })).response.status).toBe(404);
                expect((await post(`/chats/${chat.id}/summary`, {})).status).toBe(404);
                expect((await upload(chat.id, 'a.txt', Buffer.from('text'))).status).toBe(404);
                expect((await get(`/chats/${chat.id}/export?format=markdown`)).status).toBe(404);
                expect((await remove(`/chats/${chat.id}`)).status).toBe(404);
            }
            const stored = await Chat.findById(chat.id);
            expect(stored.title).toBe('Private');
            expect(stored.messages).toHaveLength(2);
        });
    });

    describe('Sending a message', () => {
        test('streams the reply, saves both messages and titles the chat', async () => {
            const chat = await newChat();
            state.chatReply = 'Hello there friend';
            const { response, events: list } = await send(chat.id, { content: 'Explain what you can help me with in three sentences.' });

            expect(response.status).toBe(200);
            expect(response.headers['content-type']).toMatch(/ndjson/);
            expect(list[0]).toMatchObject({ type: 'accepted', chat: { id: chat.id } });
            expect(list.filter((event) => event.type === 'token').map((event) => event.text).join('')).toBe('Hello there friend');
            const done = list[list.length - 1];
            expect(done).toMatchObject({ type: 'done', reply: { role: 'assistant', content: 'Hello there friend', stopped: false, error: null } });
            expect(done.reply.stats.tokensPerSecond).toBeGreaterThan(0);
            expect(done.context).toMatchObject({ capacityTokens: 32768, warn: false });

            const saved = (await get(`/chats/${chat.id}`)).data.chat;
            expect(saved.title).toBe('Explain what you can help me with in three sentences.');
            expect(saved.messages.map((message) => message.role)).toEqual(['user', 'assistant']);
            expect(saved.model).toBe('qwen2.5-coder:7b');
        });

        test('uses the default model from settings when one is set', async () => {
            await Setting.create({ defaultModel: 'custom-thing:latest' });
            const chat = await newChat();
            await send(chat.id, { content: 'hi' });
            expect(state.chatCalls[0].model).toBe('custom-thing:latest');
        });

        test('sends instructions, options, the context size and how long to stay loaded', async () => {
            await Setting.create({ keepAliveMinutes: 12 });
            const chat = await newChat({ model: 'qwen2.5-coder:7b', system: 'Reply only in formal British English.', options: { temperature: 0.3 } });
            await send(chat.id, { content: 'hi' });
            const request = state.chatCalls[0];
            expect(request.messages[0]).toEqual({ role: 'system', content: 'Reply only in formal British English.' });
            expect(request.messages[1]).toEqual({ role: 'user', content: 'hi' });
            expect(request.options).toMatchObject({ temperature: 0.3, num_ctx: 8192 });
            expect(request.keep_alive).toBe('12m');
            expect(request.think).toBeUndefined();
        });

        test('shows thinking separately for a reasoning model', async () => {
            state.thinkingModel = true;
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const { events: list } = await send(chat.id, { content: 'think hard' });
            expect(state.chatCalls[0].think).toBe(true);
            expect(list.some((event) => event.type === 'thinking')).toBe(true);
            const done = list[list.length - 1];
            expect(done.reply.thinking).toBe('Let me consider this.');
            expect(done.reply.content).not.toContain('consider');
        });

        test('a later message carries the earlier conversation', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'First answer';
            await send(chat.id, { content: 'first' });
            await send(chat.id, { content: 'second' });
            expect(state.chatCalls[1].messages.map((message) => message.content)).toEqual(['first', 'First answer', 'second']);
        });

        test('regenerate replaces the last reply', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'Old answer';
            await send(chat.id, { content: 'question' });
            state.chatReply = 'New answer';
            const { events: list } = await send(chat.id, { regenerate: true });
            expect(list[list.length - 1].reply.content).toBe('New answer');

            const saved = (await get(`/chats/${chat.id}`)).data.chat;
            expect(saved.messages.map((message) => message.content)).toEqual(['question', 'New answer']);
            expect(state.chatCalls[1].messages.map((message) => message.content)).toEqual(['question']);
        });

        test('editing an earlier message continues from the edited message', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            await send(chat.id, { content: 'one' });
            await send(chat.id, { content: 'two' });
            state.chatReply = 'Edited answer';
            await send(chat.id, { content: 'one, corrected', editIndex: 0 });

            const saved = (await get(`/chats/${chat.id}`)).data.chat;
            expect(saved.messages.map((message) => message.content)).toEqual(['one, corrected', 'Edited answer']);
        });

        test('only a user message can be edited', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            await send(chat.id, { content: 'one' });
            expect((await send(chat.id, { content: 'x', editIndex: 1 })).response.status).toBe(400);
            expect((await send(chat.id, { content: 'x', editIndex: 9 })).response.status).toBe(400);
        });

        test('refuses a message that is too long or empty', async () => {
            const chat = await newChat();
            expect((await send(chat.id, { content: 'a'.repeat(100001) })).response.status).toBe(400);
            expect((await send(chat.id, { content: '' })).response.status).toBe(400);
            expect((await Chat.findById(chat.id)).messages).toHaveLength(0);
        });

        test('refuses the 201st message', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const stored = await Chat.findById(chat.id);
            for (let count = 0; count < 100; count++) stored.messages.push({ role: 'user', content: 'q' }, { role: 'assistant', content: 'a' });
            await stored.save();
            const response = (await send(chat.id, { content: 'one more' })).response;
            expect(response.status).toBe(422);
            expect(JSON.parse(response.data)).toMatchObject({ code: 'CHAT_FULL' });
        });
    });

    describe('When something goes wrong', () => {
        test('Ollama not running is said plainly and nothing is saved', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.up = false;
            const response = (await send(chat.id, { content: 'hello' })).response;
            expect(response.status).toBe(503);
            expect(JSON.parse(response.data)).toMatchObject({ code: 'OLLAMA_DOWN' });
            expect((await Chat.findById(chat.id)).messages).toHaveLength(0);
        });

        test('no installed model is said plainly', async () => {
            state.models = [];
            const chat = await newChat();
            const response = (await send(chat.id, { content: 'hello' })).response;
            expect(response.status).toBe(409);
            expect(JSON.parse(response.data)).toMatchObject({ code: 'NO_MODELS' });
        });

        test('a model that is not installed is said plainly', async () => {
            const chat = await newChat({ model: 'gone:1b' });
            const response = (await send(chat.id, { content: 'hello' })).response;
            expect(response.status).toBe(404);
            expect(JSON.parse(response.data)).toMatchObject({ code: 'MODEL_MISSING' });
        });

        test('Ollama stopping mid-reply keeps what was written and allows regenerate', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'Partial answer that breaks';
            state.chatFailsMidway = true;
            const { events: list } = await send(chat.id, { content: 'hello' });
            const last = list[list.length - 1];
            expect(last).toMatchObject({ type: 'error', code: 'OLLAMA_DOWN' });
            expect(last.reply.content).toBe('Partial ');
            expect(last.reply.error.code).toBe('OLLAMA_DOWN');

            state.chatFailsMidway = false;
            state.chatReply = 'Whole answer';
            await send(chat.id, { regenerate: true });
            const saved = (await get(`/chats/${chat.id}`)).data.chat;
            expect(saved.messages.map((message) => message.content)).toEqual(['hello', 'Whole answer']);
        });

        test('a function call Ollama could not read is asked for once more, and said plainly when it happens twice', async () => {
            const unreadable = "invalid character '.' looking for beginning of object key string";
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatCalls.length = 0;
            state.streamErrors = [unreadable];
            state.replyPlan = ['Let me build that now', 'Here is the figure.'];
            const { events: list } = await send(chat.id, { content: 'build it' });
            // The same request goes again, the half-written first attempt is taken back, and only the second is kept.
            expect(state.chatCalls).toHaveLength(2);
            expect(state.chatCalls[1].messages).toEqual(state.chatCalls[0].messages);
            expect(list.filter((event) => event.type === 'note').map((event) => event.text)).toEqual(['The model wrote a function call that could not be read. Asking again…', '']);
            expect(list.at(-1)).toMatchObject({ type: 'done' });
            expect(list.at(-1).reply.content).toBe('Here is the figure.');
            expect(list.at(-1).reply.error).toBeNull();

            // Twice in one reply: it stops, and the person is told what happened instead of "a problem it did not explain".
            state.chatCalls.length = 0;
            state.streamErrors = [unreadable, 'error parsing tool call: raw=\'{"code": \' err=unexpected end of JSON input'];
            state.replyPlan = ['Let me build', 'Let me build'];
            const twice = await send(chat.id, { content: 'build it again' });
            expect(state.chatCalls).toHaveLength(2);
            expect(twice.events.at(-1)).toMatchObject({ type: 'error', code: 'CALL_UNREADABLE', message: 'The model tried to use a function but wrote the request so that it could not be read.' });
            state.streamErrors = [];
            state.replyPlan = [];

            // Any other failure midway is not asked again.
            state.chatCalls.length = 0;
            state.streamErrors = ['something else went wrong'];
            const other = await send(chat.id, { content: 'and again' });
            expect(state.chatCalls).toHaveLength(1);
            expect(other.events.at(-1)).toMatchObject({ type: 'error', code: 'UNKNOWN' });
            state.streamErrors = [];
        });

        test('a function called by a slightly wrong name is still run, and one that does not exist is answered with the real names', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatCalls.length = 0;
            state.toolPlan = [[['open_component_page', { component: 'Badge' }], ['Calculator', { expression: '2 + 2' }], ['fly_to_the_moon', { speed: 'fast' }]]];
            state.chatReply = 'Done.';
            const { events: list } = await send(chat.id, { content: 'try these' });
            const answers = state.chatCalls[1].messages.filter((message) => message.role === 'tool');
            expect(answers[0].content).toBe(COMPONENT_PAGES.find((page) => page.name === 'Badge').text);
            expect(answers[1].content).toMatch(/4/);
            expect(answers[2].content).toMatch(/^There is no function called fly_to_the_moon\. The functions you can call are: open_component, open_skill_file, open_skill, find_skill, ask_user, save_note, list_my_files, read_my_file, save_to_files, date_time, calculator, read_web_page\.$/);
            expect(list.at(-1).reply.steps.map(({ tool, failed }) => ({ tool, failed: failed || null }))).toEqual([{ tool: 'Component', failed: null }, { tool: 'Calculator', failed: null }, { tool: 'fly_to_the_moon', failed: 'failed' }]);
            state.toolPlan = [];
        });

        test('a model can put a question to the person with choices to press, and that ends the reply', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatCalls.length = 0;
            state.toolsSeen = [];
            state.toolPlan = [[['ask_user', { question: 'Which engine do you mean?', choices: ['Petrol, four-stroke', 'Diesel', 'Petrol, four-stroke', 'A <jet> "turbine"', '', 'Rotary', 'Steam'] }]]];
            state.chatReply = 'Happy to help.';
            const { events: list } = await send(chat.id, { content: 'show me how an engine works' });
            const asker = state.toolsSeen[0].find((tool) => tool.function.name === 'ask_user');
            expect(asker.function.description).toMatch(/^Ask the user one question and end your reply, when the request leaves open something that would change the answer/);
            expect(asker.function.description).toMatch(/Do not ask what you can settle yourself, look up, or reasonably assume/);
            // The model is not asked to go on: one request, and the reply ends with the question and its buttons.
            expect(state.chatCalls).toHaveLength(1);
            const reply = list.at(-1).reply;
            expect(list.at(-1)).toMatchObject({ type: 'done' });
            expect(reply.content.trim().endsWith(['Which engine do you mean?', '', '```component', '<ButtonGroup>', '<Button send="Petrol, four-stroke" size="sm">Petrol, four-stroke</Button>', '<Button send="Diesel" size="sm">Diesel</Button>', '<Button send="A jet turbine" size="sm">A jet turbine</Button>', '<Button send="Rotary" size="sm">Rotary</Button>', '</ButtonGroup>', '```'].join('\n'))).toBe(true);
            expect(componentFaults(reply.content).faults).toEqual([]);
            expect(reply.steps.map(({ tool, input, failed }) => ({ tool, input, failed: failed || null }))).toEqual([{ tool: 'Question', input: 'Which engine do you mean?', failed: null }]);
            expect(list.filter((event) => event.type === 'note').map((event) => event.text)).toEqual(['Asking you a question…', '']);

            // With no choices it is the question alone; with no question nothing is asked and the model goes on.
            state.chatCalls.length = 0;
            state.toolPlan = [[['ask_user', { question: 'What is it for?' }]]];
            const bare = await send(chat.id, { content: 'and again' });
            expect(bare.events.at(-1).reply.content.trim().endsWith('What is it for?')).toBe(true);
            state.chatCalls.length = 0;
            state.toolPlan = [[['ask_user', { question: '  ' }]]];
            const none = await send(chat.id, { content: 'once more' });
            expect(state.chatCalls).toHaveLength(2);
            expect(state.chatCalls[1].messages.at(-1)).toMatchObject({ role: 'tool', content: 'Nothing was asked. Give the question.' });
            expect(none.events.at(-1).reply.content).toBe('Happy to help.');
            state.toolPlan = [];
        });

        test('not enough memory is said plainly', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatError = 'model requires more system memory (9.1 GiB) than is available (4.0 GiB)';
            const { events: list } = await send(chat.id, { content: 'hello' });
            expect(list[list.length - 1]).toMatchObject({ type: 'error', code: 'OUT_OF_MEMORY' });
            expect(list[list.length - 1].action).toMatch(/unload/i);
        });

        test('a conversation too long for the model is refused before it is sent', async () => {
            state.contextLength = 2048;
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const response = (await send(chat.id, { content: 'word '.repeat(2000) })).response;
            expect(response.status).toBe(422);
            expect(JSON.parse(response.data)).toMatchObject({ code: 'TOO_LONG' });
            expect(state.chatCalls).toHaveLength(0);
        });

        test('warns as the conversation nears the limit', async () => {
            state.contextLength = 8192;
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const { events: list } = await send(chat.id, { content: 'word '.repeat(Math.floor(((8192 - 1024 - 300 - estimateTokens(COMPONENT_GUIDE) - estimateTokens(SKILL_MAP)) * 4) / 5)) });
            expect(list[list.length - 1].context).toMatchObject({ capacityTokens: 8192, warn: true });
            expect(state.chatCalls[0].options.num_ctx).toBe(8192);
        });
    });

    describe('Stopping and waiting', () => {
        const openStream = async (chatId, body) => {
            const controller = new AbortController();
            const cookie = client.cookieJar.getCookieStringSync(client.baseURL);
            const response = await fetch(`${client.baseURL}/api/v1/chats/${chatId}/messages`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', Cookie: cookie, 'X-CSRF-Token': client.getCsrfTokenFromCookies() },
                body: JSON.stringify(body),
                signal: controller.signal
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            const seen = [];
            const pump = (async () => {
                let buffer = '';
                try {
                    for (;;) {
                        const { value, done } = await reader.read();
                        if (done) break;
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop();
                        lines.filter(Boolean).forEach((line) => seen.push(JSON.parse(line)));
                    }
                } catch {
                    return seen;
                }
                return seen;
            })();
            const until = async (check) => {
                for (let attempt = 0; attempt < 200; attempt++) {
                    if (seen.some(check)) return;
                    await wait(15);
                }
                throw new Error(`stream never showed the expected event: ${JSON.stringify(seen)}`);
            };
            return { seen, until, stop: () => controller.abort(), finished: pump };
        };

        test('leaving the page does not stop the reply, and coming back picks it up from the start', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'One two three four';
            state.chatHold = true;
            const first = await openStream(chat.id, { content: 'long request' });
            await first.until((event) => event.type === 'token');
            first.stop();
            await first.finished;
            await wait(60);

            expect(state.chatAborted).toBe(0);
            expect((await get(`/chats/${chat.id}`)).data.chat).toMatchObject({ replying: true });
            expect((await post(`/chats/${chat.id}/messages`, { regenerate: true })).status).toBe(409);

            const cookie = client.cookieJar.getCookieStringSync(client.baseURL);
            const back = await fetch(`${client.baseURL}/api/v1/chats/${chat.id}/reply`, { headers: { Cookie: cookie } });
            expect(back.status).toBe(200);
            state.chatHold = false;
            const events = (await back.text()).split('\n').filter(Boolean).map((line) => JSON.parse(line));
            expect(events.slice(0, 2).map((event) => event.type)).toEqual(['accepted', 'start']);
            expect(events.filter((event) => event.type === 'token').map((event) => event.text).join('')).toBe('One two three four');
            expect(events[events.length - 1]).toMatchObject({ type: 'done', reply: { content: 'One two three four', stopped: false } });

            const after = (await get(`/chats/${chat.id}`)).data.chat;
            expect(after.replying).toBe(false);
            expect(after.messages[1]).toMatchObject({ role: 'assistant', content: 'One two three four' });
            expect((await fetch(`${client.baseURL}/api/v1/chats/${chat.id}/reply`, { headers: { Cookie: cookie } })).status).toBe(204);
        });

        test('a reply sent live answers at once and goes out on the live connection, numbered and in order', async () => {
            const service = getNotificationService();
            const sent = [];
            const real = service.sendToUser.bind(service);
            service.sendToUser = (userId, notification) => {
                if (notification.type === 'chat:reply') sent.push({ userId: String(userId), ...notification.data });
                return real(userId, notification);
            };
            try {
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                expect((await get(`/chats/${chat.id}/live`)).data).toMatchObject({ replying: false, seq: 0, events: [] });

                const started = await post(`/chats/${chat.id}/messages`, { content: 'say something', live: true });
                expect(started.status).toBe(202);
                expect(started.data).toEqual({ success: true, started: true });

                for (let attempt = 0; attempt < 200 && !sent.some((piece) => piece.event.type === 'end'); attempt++) await wait(25);
                const mine = sent.filter((piece) => piece.chatId === chat.id);
                expect(mine.map((piece) => piece.seq)).toEqual(mine.map((_, index) => index + 1));
                expect(mine[0].event.type).toBe('accepted');
                expect(mine.some((piece) => piece.event.type === 'token')).toBe(true);
                expect(mine[mine.length - 1].event).toEqual({ type: 'end' });
                expect(new Set(mine.map((piece) => piece.userId)).size).toBe(1);
                expect(mine[0].userId).toBe(String((await Chat.findById(chat.id)).owner));

                for (let attempt = 0; attempt < 100 && (await get(`/chats/${chat.id}/live`)).data.replying; attempt++) await wait(25);
                const saved = (await get(`/chats/${chat.id}`)).data.chat;
                expect(saved.messages[saved.messages.length - 1].role).toBe('assistant');
                expect(saved.messages[saved.messages.length - 1].content.length).toBeGreaterThan(0);

                const refused = await post(`/chats/${chat.id}/messages`, { live: true });
                expect(refused.status).toBe(400);
            } finally {
                service.sendToUser = real;
            }
        });

        test('a change to a chat is announced to the person\'s open apps, and a read is not', async () => {
            const service = getNotificationService();
            const sent = [];
            const real = service.sendToUser.bind(service);
            service.sendToUser = (userId, notification) => {
                if (['chat:changed', 'models:changed', 'device:changed'].includes(notification.type)) sent.push({ type: notification.type, ...notification.data });
                return real(userId, notification);
            };
            const settled = async (count) => {
                for (let attempt = 0; attempt < 80 && sent.length < count; attempt++) await wait(10);
            };
            try {
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                await settled(1);
                expect(sent).toEqual([{ type: 'chat:changed', id: null, removed: false }]);

                sent.length = 0;
                await get(`/chats/${chat.id}`);
                await get('/chats');
                await wait(60);
                expect(sent).toEqual([]);

                await client.patch(`/api/v1/chats/${chat.id}`, { title: 'Renamed elsewhere' });
                await settled(1);
                expect(sent).toEqual([{ type: 'chat:changed', id: chat.id, removed: false }]);

                sent.length = 0;
                expect((await client.patch(`/api/v1/chats/${chat.id}`, { title: '' })).status).toBe(400);
                await wait(60);
                expect(sent).toEqual([]);

                await send(chat.id, { content: 'hello' });
                await settled(2);
                expect(sent.filter((note) => note.type === 'chat:changed' && note.id === chat.id).length).toBeGreaterThanOrEqual(2);

                sent.length = 0;
                await client.delete(`/api/v1/chats/${chat.id}`);
                await settled(1);
                expect(sent).toEqual([{ type: 'chat:changed', id: chat.id, removed: true }]);

                sent.length = 0;
                await post('/models/load', { name: 'qwen2.5-coder:7b' });
                await wait(60);
                expect(sent.some((note) => note.type === 'models:changed')).toBe(false);
                await testStartup.loginAsUser('admin');
                await post('/models/unload', { name: 'qwen2.5-coder:7b' });
                await settled(1);
                expect(sent.some((note) => note.type === 'models:changed')).toBe(true);
            } finally {
                service.sendToUser = real;
            }
        });

        test('an app that joins a reply part-way gets what was sent so far and the number to carry on from', async () => {
            const service = getNotificationService();
            const sent = [];
            const real = service.sendToUser.bind(service);
            service.sendToUser = (userId, notification) => {
                if (notification.type === 'chat:reply') sent.push(notification.data);
                return real(userId, notification);
            };
            try {
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                state.chatReply = 'One two three four';
                state.chatHold = true;
                expect((await post(`/chats/${chat.id}/messages`, { content: 'count for me', live: true })).status).toBe(202);
                for (let attempt = 0; attempt < 200 && !sent.some((piece) => piece.chatId === chat.id && piece.event.type === 'token'); attempt++) await wait(15);

                const joined = (await get(`/chats/${chat.id}/live`)).data;
                expect(joined.replying).toBe(true);
                const before = sent.filter((piece) => piece.chatId === chat.id && piece.seq <= joined.seq);
                expect(joined.seq).toBe(before.length);
                const text = (list) => list.filter((event) => event.type === 'token').map((event) => event.text).join('');
                expect(text(joined.events)).toBe(text(before.map((piece) => piece.event)));

                state.chatHold = false;
                for (let attempt = 0; attempt < 300 && !sent.some((piece) => piece.chatId === chat.id && piece.event.type === 'end'); attempt++) await wait(15);
                const later = sent.filter((piece) => piece.chatId === chat.id && piece.seq > joined.seq);
                expect(later.map((piece) => piece.seq)).toEqual(later.map((_, index) => joined.seq + 1 + index));
                expect(text(joined.events) + text(later.map((piece) => piece.event))).toBe('One two three four');
                expect(later[later.length - 1].event).toEqual({ type: 'end' });

                const other = client.authenticatedAs;
                await testStartup.loginAsUser(other === 'admin' ? 'user' : 'admin');
                expect((await get(`/chats/${chat.id}/live`)).status).toBe(404);
                await testStartup.loginAsUser(other);
            } finally {
                state.chatHold = false;
                service.sendToUser = real;
            }
        });

        test('activity lists your own replies in progress and the downloads, and nobody else\'s replies', async () => {
            expect((await get('/activity')).data).toMatchObject({ replies: [], downloads: expect.any(Array) });
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'One two three four';
            state.chatHold = true;
            const stream = await openStream(chat.id, { content: 'a long request about ducks' });
            await stream.until((event) => event.type === 'token');
            await post(`/chats/${chat.id}/messages`, { content: 'and then geese' });

            const mine = (await get('/activity')).data;
            expect(mine.replies).toEqual([{ chatId: chat.id, title: 'a long request about ducks', model: 'qwen2.5-coder:7b', waiting: 1, phase: expect.stringMatching(/^(starting|waiting|thinking|searching|reading|writing)$/), words: expect.any(Number) }]);

            const owner = client.authenticatedAs;
            await testStartup.loginAsUser(owner === 'admin' ? 'user' : 'admin');
            expect((await get('/activity')).data.replies).toEqual([]);
            await testStartup.loginAsUser(owner || 'user');

            await post(`/chats/${chat.id}/stop`, {});
            state.chatHold = false;
            await stream.finished;
            for (let attempt = 0; attempt < 100; attempt++) {
                if ((await get('/activity')).data.replies.length === 0) break;
                await wait(15);
            }
            expect((await get('/activity')).data.replies).toEqual([]);

            client.clearCookies();
            client.authenticatedAs = null;
            expect((await get('/activity')).status).toBe(401);
        });

        test('Stop ends a reply at once while its model is still loading into memory', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.loadHold = true;
            state.loadAborted = 0;
            try {
                const stream = await openStream(chat.id, { content: 'hello' });
                await stream.until((event) => event.type === 'start');
                for (let attempt = 0; attempt < 100 && state.chatCalls.length === 0; attempt++) await wait(15);
                const asked = Date.now();
                expect((await post(`/chats/${chat.id}/stop`, {})).data).toMatchObject({ stopped: true });
                await stream.finished;
                expect(Date.now() - asked).toBeLessThan(2000);
                expect(state.loadAborted).toBe(1);
                expect(stream.seen.at(-1).type).toBe('done');
                expect(stream.seen.some((event) => event.type === 'error')).toBe(false);
                for (let attempt = 0; attempt < 100; attempt++) {
                    if ((await get('/activity')).data.replies.length === 0) break;
                    await wait(15);
                }
                const kept = await Chat.findById(chat.id);
                expect(kept.messages.map((message) => message.role)).toEqual(['user']);
            } finally {
                state.loadHold = false;
            }
        });

        test('a chat sent to a computer is ended through its own request, before the computer has answered', async () => {
            // The path real computers use: the Ollama library over the link to a desktop app. Here the link never answers.
            const asked = [];
            const link = (input, init = {}) => new Promise((resolve, reject) => {
                asked.push(init.signal);
                init.signal?.addEventListener('abort', () => reject(new DOMException('This operation was aborted', 'AbortError')), { once: true });
            });
            const halt = new AbortController();
            const waiting = createOllamaClient('http://device.mollama', link).chat({ model: 'small:1b', messages: [{ role: 'user', content: 'hi' }], stream: true }, halt.signal);
            let ended = null;
            waiting.then(() => { ended = 'answered'; }, () => { ended = 'stopped'; });
            await wait(50);
            expect(ended).toBeNull();
            expect(asked).toHaveLength(1);
            expect(asked[0].aborted).toBe(false);
            halt.abort();
            await waiting.catch(() => {});
            expect(ended).toBe('stopped');
            expect(asked[0].aborted).toBe(true);
        });

        test('Stop does not wait for a tool that is still running', async () => {
            process.env.TAVILY_API_KEY = 'tvly-test';
            {
                // A search that never comes back: only Stop can end this reply.
                const usual = setSearchFetch(() => new Promise(() => {}));
                try {
                    state.toolPlan = [[['web_search', { query: 'duck 3d model' }]]];
                    const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                    const stream = await openStream(chat.id, { content: 'find a duck model' });
                    await stream.until((event) => event.type === 'tool');
                    const asked = Date.now();
                    expect((await post(`/chats/${chat.id}/stop`, {})).data).toMatchObject({ stopped: true });
                    await stream.finished;
                    expect(Date.now() - asked).toBeLessThan(2000);
                    expect(stream.seen.at(-1).type).toBe('done');
                    // The model is not asked again once the reply was stopped.
                    expect(state.chatCalls).toHaveLength(1);
                    for (let attempt = 0; attempt < 100; attempt++) {
                        if ((await get('/activity')).data.replies.length === 0) break;
                        await wait(15);
                    }
                    expect((await get('/activity')).data.replies).toEqual([]);
                } finally {
                    setSearchFetch(usual);
                    process.env.TAVILY_API_KEY = '';
                }
            }
        });

        test('the skill map is sent with every chat, and its worked examples are opened one at a time', async () => {
            const map = SKILLS.find((skill) => skill.id === 'skill-map');
            expect(map).toMatchObject({ source: 'mollama', default: true });
            expect(map.files.map((file) => file.path)).toEqual(['examples/combining.md', 'examples/comparison.md', 'examples/concept.md', 'examples/machine.md', 'examples/plain.md', 'examples/process.md']);
            // Every example the map names is one of its files, and every component an example draws can be drawn.
            for (const file of map.files) {
                expect(map.text).toContain(file.path);
                expect(componentFaults(fs.readFileSync(`${map.folder}/${file.path}`, 'utf8')).faults).toEqual([]);
            }
            state.mapsSeen = 0;
            state.toolPlan = [[['open_skill_file', { skill: 'skill-map', file: 'examples/machine.md' }]]];
            state.chatReply = 'Here is how it is built.';
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            await send(chat.id, { content: 'how is an electric bike built?' });
            expect(state.mapsSeen).toBeGreaterThan(0);
            expect(state.toolsSeen[0].find((tool) => tool.function.name === 'open_skill_file').function.parameters.properties.skill.enum).toEqual(['skill-map']);
            expect(state.chatCalls[1].messages.at(-1)).toMatchObject({ role: 'tool', content: expect.stringMatching(/^EXAMPLE: HOW A MACHINE IS BUILT/) });
            state.toolPlan = [];
        });

        test('a reply cut short for want of room is tried again with more, then carried on until it is finished', async () => {
            // Nothing caps how much a model writes unless the person set a limit for the chat.
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatCalls.length = 0;
            await send(chat.id, { content: 'hello' });
            const usual = state.chatCalls[0].options.num_ctx;
            expect(usual).toBeGreaterThanOrEqual(8192);
            expect(state.chatCalls[0].options.num_predict).toBe(-1);

            // Cut once: the same question is asked again with the next size up, and only the full answer is kept.
            state.chatCalls.length = 0;
            state.endings = ['length', 'stop'];
            state.replyPlan = ['Half an', 'The whole answer.'];
            const { events: list } = await send(chat.id, { content: 'a long one' });
            expect(state.chatCalls).toHaveLength(2);
            expect(state.chatCalls[1].options.num_ctx).toBeGreaterThan(state.chatCalls[0].options.num_ctx);
            expect(list.at(-1)).toMatchObject({ type: 'done' });
            expect(list.at(-1).reply.content).toBe('The whole answer.');
            expect(list.at(-1).reply.error).toBeNull();
            expect(list.filter((event) => event.type === 'note').map((event) => event.text)).toEqual(['Giving the model more room and trying again…', '']);

            // Cut again with nothing more to give: the model is asked to go on from where it stopped, as often as it
            // takes, and the pieces make one reply.
            state.chatCalls.length = 0;
            state.endings = ['length', 'length', 'length', 'stop'];
            state.replyPlan = ['Half an', 'Half an answer,', 'then more,', 'and the end.'];
            const carried = await send(chat.id, { content: 'another long one' });
            expect(state.chatCalls).toHaveLength(4);
            expect(state.chatCalls[2].messages.at(-1)).toMatchObject({ role: 'user', content: expect.stringContaining('Go on from exactly where you stopped') });
            expect(state.chatCalls[2].messages.at(-2)).toMatchObject({ role: 'assistant', content: 'Half an answer,' });
            expect(carried.events.at(-1)).toMatchObject({ type: 'done' });
            expect(carried.events.at(-1).reply.content).toBe('Half an answer,then more,and the end.');
            expect(carried.events.at(-1).reply.error).toBeNull();
            expect((await get(`/chats/${chat.id}`)).data.chat.messages.at(-1)).toMatchObject({ role: 'assistant', content: 'Half an answer,then more,and the end.' });

            // A reply that got no further than thinking is kept and explained, not dropped.
            state.chatCalls.length = 0;
            state.endings = ['length', 'length'];
            state.replyPlan = ['', ''];
            const nothing = await send(chat.id, { content: 'and another' });
            expect(nothing.events.at(-1)).toMatchObject({ type: 'error', code: 'REPLY_CUT' });
            expect((await get(`/chats/${chat.id}`)).data.chat.messages.at(-1)).toMatchObject({ role: 'assistant', content: '', error: { code: 'REPLY_CUT' } });
            state.endings = [];
            state.replyPlan = [];
        });

        test('Stop ends generation and keeps what was written', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'One two three four';
            state.chatHold = true;
            const stream = await openStream(chat.id, { content: 'long request' });
            await stream.until((event) => event.type === 'token');
            expect((await post(`/chats/${chat.id}/stop`, {})).data).toMatchObject({ stopped: true });
            await stream.finished;
            expect((await post(`/chats/${chat.id}/stop`, {})).data).toMatchObject({ stopped: false });

            for (let attempt = 0; attempt < 100 && state.chatAborted === 0; attempt++) await wait(15);
            expect(state.chatAborted).toBe(1);
            for (let attempt = 0; attempt < 100; attempt++) {
                if ((await Chat.findById(chat.id)).messages.length === 2) break;
                await wait(15);
            }
            const stored = await Chat.findById(chat.id);
            expect(stored.messages[1]).toMatchObject({ role: 'assistant', content: 'One ', stopped: true });

            state.chatHold = false;
            state.chatReply = 'Fresh answer';
            const { events: list } = await send(chat.id, { regenerate: true });
            expect(list[list.length - 1].reply.content).toBe('Fresh answer');
        });

        test('a second request waits and is told its place, then answers', async () => {
            const first = await newChat({ model: 'qwen2.5-coder:7b' });
            const second = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatHold = true;
            const one = await openStream(first.id, { content: 'first long request' });
            await one.until((event) => event.type === 'token');
            const two = await openStream(second.id, { content: 'second long request' });
            await two.until((event) => event.type === 'queued');
            expect(two.seen.find((event) => event.type === 'queued')).toEqual({ type: 'queued', ahead: 1 });
            expect(two.seen.some((event) => event.type === 'start')).toBe(false);

            state.chatHold = false;
            const [firstEvents, secondEvents] = await Promise.all([one.finished, two.finished]);
            expect(firstEvents[firstEvents.length - 1].type).toBe('done');
            expect(secondEvents[secondEvents.length - 1].type).toBe('done');
            expect(secondEvents.findIndex((event) => event.type === 'start')).toBeGreaterThan(secondEvents.findIndex((event) => event.type === 'queued'));
        });

        test('a waiting request that is stopped never reaches the model', async () => {
            const first = await newChat({ model: 'qwen2.5-coder:7b' });
            const second = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatHold = true;
            const one = await openStream(first.id, { content: 'first' });
            await one.until((event) => event.type === 'token');
            const two = await openStream(second.id, { content: 'second' });
            await two.until((event) => event.type === 'queued');
            await post(`/chats/${second.id}/stop`, {});
            await two.finished;
            await wait(50);

            state.chatHold = false;
            await one.finished;
            await wait(50);
            expect(state.chatCalls).toHaveLength(1);
            expect((await Chat.findById(second.id)).messages.map((message) => message.role)).toEqual(['user']);
        });

        test('a message sent while the chat is replying waits and is answered next', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatReply = 'Answer';
            state.chatHold = true;
            const one = await openStream(chat.id, { content: 'first' });
            await one.until((event) => event.type === 'token');
            const queued = await post(`/chats/${chat.id}/messages`, { content: 'second' });
            expect(queued.status).toBe(202);
            expect(queued.data.queued).toEqual(['second']);
            expect((await get(`/chats/${chat.id}`)).data.chat.queued).toEqual(['second']);
            expect((await post(`/chats/${chat.id}/messages`, { regenerate: true })).status).toBe(409);

            state.chatHold = false;
            await one.finished;
            for (let attempt = 0; attempt < 200; attempt++) {
                if ((await Chat.findById(chat.id)).messages.length === 4) break;
                await wait(15);
            }
            const stored = await Chat.findById(chat.id);
            expect(stored.messages.map((message) => message.role)).toEqual(['user', 'assistant', 'user', 'assistant']);
            expect(stored.messages[2].content).toBe('second');
            expect(stored.messages[3].content).toBe('Answer');
            for (let attempt = 0; attempt < 100; attempt++) {
                if (!(await get(`/chats/${chat.id}`)).data.chat.replying) break;
                await wait(15);
            }
            expect((await get(`/chats/${chat.id}`)).data.chat).toMatchObject({ replying: false, queued: [] });
        });

        test('only five messages can wait in one chat, and Stop drops the ones waiting', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatHold = true;
            const one = await openStream(chat.id, { content: 'first' });
            await one.until((event) => event.type === 'token');
            for (let index = 0; index < 5; index++) {
                expect((await post(`/chats/${chat.id}/messages`, { content: `waiting ${index}` })).status).toBe(202);
            }
            const full = await post(`/chats/${chat.id}/messages`, { content: 'one too many' });
            expect(full.status).toBe(429);
            expect(full.data).toMatchObject({ code: 'QUEUE_FULL' });

            await post(`/chats/${chat.id}/stop`, {});
            await one.finished;
            state.chatHold = false;
            await wait(150);
            expect(state.chatCalls).toHaveLength(1);
            expect((await Chat.findById(chat.id)).messages.map((message) => message.role)).toEqual(['user', 'assistant']);
            expect((await get(`/chats/${chat.id}`)).data.chat).toMatchObject({ replying: false, queued: [] });
        });

        test('the number of generations at once follows the setting', async () => {
            await Setting.create({ maxConcurrentGenerations: 2 });
            const first = await newChat({ model: 'qwen2.5-coder:7b' });
            const second = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatHold = true;
            const one = await openStream(first.id, { content: 'first' });
            const two = await openStream(second.id, { content: 'second' });
            await one.until((event) => event.type === 'token');
            await two.until((event) => event.type === 'token');
            expect(two.seen.some((event) => event.type === 'queued')).toBe(false);
            state.chatHold = false;
            await Promise.all([one.finished, two.finished]);
        });
    });

    describe('Attachments', () => {
        test('an image is kept as it is for a model that can see, sent with the next message, and refused for one that cannot', async () => {
            const png = Buffer.concat([Buffer.from('89504e470d0a1a0a0000000d49484452', 'hex'), Buffer.alloc(200, 7)]);
            state.vision = true;
            try {
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const attached = await upload(chat.id, 'photo.png', png);
                expect(attached.status).toBe(201);
                expect(attached.data.chat.pendingAttachments[0]).toMatchObject({ name: 'photo.png', image: true, chars: 0, tokens: 1000 });
                expect(JSON.stringify(attached.data)).not.toContain(png.toString('base64'));

                state.chatCalls.length = 0;
                await send(chat.id, { content: 'what is this?' });
                const sentNow = state.chatCalls[0].messages.at(-1);
                expect(sentNow.images).toEqual([png.toString('base64')]);
                expect(sentNow.content).toBe('what is this?');

                // Too large to keep with a chat.
                const large = await upload(chat.id, 'big.jpg', Buffer.alloc(4 * 1024 * 1024 + 1, 1));
                expect(large.status).toBe(413);
                expect(large.data.code).toBe('IMAGE_TOO_LARGE');

                // A model that cannot see is not sent images, and a new one is refused.
                state.vision = false;
                state.chatCalls.length = 0;
                await send(chat.id, { content: 'and now?' });
                expect(state.chatCalls[0].messages.some((message) => message.images)).toBe(false);
                const refused = await upload(chat.id, 'photo.png', png);
                expect(refused.status).toBe(400);
                expect(refused.data).toMatchObject({ code: 'NO_VISION', message: 'This model cannot see images.' });
            } finally {
                state.vision = false;
            }
        });

        test.each([
            ['notes.txt', () => Buffer.from('The harbour is twelve metres deep.'), 'twelve metres'],
            ['figures.csv', () => Buffer.from('port,depth\nHarwich,12'), 'Harwich,12'],
            ['report.pdf', () => makePdf('Dredging costs rose in March'), 'Dredging costs rose'],
            ['minutes.docx', () => makeDocx('The board approved the survey'), 'board approved']
        ])('%s is read and sent with the next message only', async (name, build, expected) => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const attached = await upload(chat.id, name, await build());
            expect(attached.status).toBe(201);
            expect(attached.data.chat.pendingAttachments).toHaveLength(1);
            expect(attached.data.chat.pendingAttachments[0]).toMatchObject({ name, source: 'ai-base' });
            expect(attached.data.chat.pendingAttachments[0].text).toBeUndefined();
            expect(attached.data.chat.context.usedTokens).toBeGreaterThan(0);

            await send(chat.id, { content: 'Summarise this.' });
            expect(state.chatCalls[0].messages[0].content).toContain(expected);
            expect(state.chatCalls[0].messages[0].content).toContain(`Document "${name}"`);

            const saved = (await get(`/chats/${chat.id}`)).data.chat;
            expect(saved.pendingAttachments).toEqual([]);
            expect(saved.messages[0].attachments[0].name).toBe(name);
            expect(saved.messages[0].content).toBe('Summarise this.');

            await send(chat.id, { content: 'And again?' });
            expect(state.chatCalls[1].messages[0].content).toContain(expected);
            expect(state.chatCalls[1].messages[2].content).toBe('And again?');
        });

        test('a file that is not text is refused in plain words', async () => {
            const chat = await newChat();
            const response = await upload(chat.id, 'bundle.zip', Buffer.from([0x50, 0x4b, 3, 4, 0, 0, 0, 13, 0, 1, 2, 0xff]));
            expect(response.status).toBe(415);
            expect(response.data).toMatchObject({ success: false, code: 'FILE_UNREADABLE' });
        });

        test('a file over 20 MB is refused', async () => {
            const chat = await newChat();
            const response = await upload(chat.id, 'huge.txt', Buffer.alloc(MAX_ATTACHMENT_BYTES + 1024, 'a'));
            expect(response.status).toBe(413);
            expect(response.data).toMatchObject({ code: 'FILE_TOO_LARGE' });
            expect((await Chat.findById(chat.id)).attachments).toHaveLength(0);
        }, 30000);

        test('a document longer than the model can read is refused with the numbers', async () => {
            state.contextLength = 4096;
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const response = await upload(chat.id, 'long.txt', Buffer.from('word '.repeat(8000)));
            expect(response.status).toBe(422);
            expect(response.data).toMatchObject({ code: 'DOCUMENT_TOO_LONG', neededTokens: 10000 });
            expect(response.data.message).toMatch(/10,000 tokens/);
            expect(response.data.action).toMatch(/shorter document|read more/);
            expect((await Chat.findById(chat.id)).attachments).toHaveLength(0);
        });

        test('needs a file name', async () => {
            const chat = await newChat();
            const response = await post(`/chats/${chat.id}/attachments`, Buffer.from('text'), { headers: { 'Content-Type': 'application/octet-stream' } });
            expect(response.status).toBe(400);
        });

        test('an attachment can be removed before it is sent, not after', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            const attached = (await upload(chat.id, 'a.txt', Buffer.from('alpha'))).data.chat.pendingAttachments[0];
            expect((await remove(`/chats/${chat.id}/attachments/${attached.id}`)).data.chat.pendingAttachments).toEqual([]);

            const kept = (await upload(chat.id, 'b.txt', Buffer.from('beta'))).data.chat.pendingAttachments[0];
            await send(chat.id, { content: 'use it' });
            expect((await remove(`/chats/${chat.id}/attachments/${kept.id}`)).status).toBe(404);
        });

        test('attachments go when the chat is deleted', async () => {
            const chat = await newChat();
            await upload(chat.id, 'a.txt', Buffer.from('alpha'));
            await remove(`/chats/${chat.id}`);
            expect(await Chat.countDocuments({ 'attachments.name': 'a.txt' })).toBe(0);
        });
    });

    describe('Export and summary', () => {
        test('exports as markdown and as JSONL', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', title: 'Harbour: plan/2026', system: 'Be brief.' });
            state.chatReply = 'It is deep.';
            await send(chat.id, { content: 'How deep is it?' });

            const markdown = (await get(`/chats/${chat.id}/export?format=markdown`)).data;
            expect(markdown.fileName).toBe('harbour-plan-2026.md');
            expect(markdown.content).toContain('# Harbour: plan/2026');
            expect(markdown.content).toContain('## You\n\nHow deep is it?');
            expect(markdown.content).toContain('It is deep.');

            const jsonl = (await get(`/chats/${chat.id}/export?format=jsonl`)).data;
            expect(jsonl.fileName).toBe('harbour-plan-2026.jsonl');
            const lines = jsonl.content.trim().split('\n').map((line) => JSON.parse(line));
            expect(lines.map((line) => line.role)).toEqual(['system', 'user', 'assistant']);
            expect(lines[2]).toMatchObject({ content: 'It is deep.', model: 'qwen2.5-coder:7b' });

            expect((await get(`/chats/${chat.id}/export?format=pdf`)).status).toBe(400);
        });

        test('starts a new chat from a summary', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', system: 'Be brief.' });
            await send(chat.id, { content: 'We plan a harbour survey.' });
            state.chatReply = '- Goal: harbour survey';
            const response = await post(`/chats/${chat.id}/summary`, {});

            expect(response.status).toBe(201);
            expect(response.data.chat).toMatchObject({
                summary: '- Goal: harbour survey', continuedFrom: chat.id, model: 'qwen2.5-coder:7b', system: 'Be brief.', messages: []
            });
            expect(response.data.chat.title).toMatch(/\(continued\)$/);
            expect(state.chatCalls[1].stream).toBe(false);

            await send(response.data.chat.id, { content: 'What was the goal?' });
            expect(state.chatCalls[2].messages[0].content).toContain('Summary of the earlier conversation:\n- Goal: harbour survey');
        });

        test('an empty chat cannot be summarised', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            expect((await post(`/chats/${chat.id}/summary`, {})).status).toBe(400);
        });
    });

    // The components the chat can draw and the attributes it accepts for each, as in the web app's own list.
    const BOX_ATTRIBUTES = ['layout', 'columns', 'gap', 'padding', 'justify', 'align', 'wrap', 'flexFill', 'width', 'height', 'minWidth', 'minHeight', 'maxWidth', 'backgroundColor', 'background', 'backgroundTone', 'backgroundIntensity', 'backgroundSize', 'backgroundMotion', 'backgroundFade'];
    const BLOCK_ATTRIBUTES = {
        Container: [...BOX_ATTRIBUTES, 'hoverable'],
        Card: [...BOX_ATTRIBUTES, 'bordered', 'borderColor', 'borderStrength', 'radius', 'shadow', 'hover'],
        Divider: ['label', 'color', 'variant', 'size'],
        Typography: ['size', 'weight', 'color', 'font', 'as', 'href', 'truncate'],
        Badge: ['color', 'size'],
        Icon: ['name', 'color', 'size'],
        Button: ['color', 'variant', 'size', 'icon', 'send', 'disabled', 'width', 'align', 'flat'],
        ButtonGroup: [],
        FloatingActionButton: ['icon', 'color', 'size', 'position', 'badge', 'badgeColor', 'draggable', 'send'],
        ProgressBar: ['value', 'max', 'label', 'showPercentage', 'color', 'size', 'striped', 'animated'],
        CircularProgress: ['size', 'color'],
        Data: ['src', 'variant', 'key', 'label', 'as', 'colors'],
        Flow: ['height', 'arrange', 'minimap', 'background', 'id', 'label', 'color', 'shape', 'size', 'weight', 'align', 'width', 'x', 'y', 'group', 'source', 'target', 'type', 'arrow', 'start', 'dashed', 'animated', 'labelsize', 'labelbg', 'from', 'to', 'bend', 'corner', 'collapsible', 'collapsed'],
        TreeView: ['label'],
        Image: ['src', 'alt', 'width', 'height'],
        Video: ['src', 'title'],
        Audio: ['src', 'title'],
        PdfViewer: ['src'],
        Slides: ['src', 'height'],
        Sheet: ['src', 'height'],
        Model3D: ['src', 'format', 'fileName', 'autoRotate', 'height'],
        Editor: ['src', 'mode', 'language'],
        Input: ['label', 'placeholder', 'multiline', 'rows'],
        Select: ['label', 'placeholder', 'value'],
        Switch: ['label', 'defaultChecked'],
        Frame: ['height', 'title', 'template', 'palette', 'name', 'file']
    };
    const BLOCK_NAMES = Object.keys(BLOCK_ATTRIBUTES);

    describe('Built-in component guide', () => {
        test('the index says how to write a component and lists them all, and tells no model when to use one', () => {
            expect(COMPONENT_GUIDE).toContain('\nHOW TO DRAW\n');
            expect(COMPONENT_GUIDE).toContain('```component\n<Card ');
            expect(COMPONENT_GUIDE).toMatch(/Whether and where a reply uses them is yours to judge/);
            expect(COMPONENT_GUIDE.split('\nTHE COMPONENTS\n')[1].split('\n').filter((line) => line.startsWith('- ')).map((line) => line.slice(2).split(':')[0])).toEqual(BLOCK_NAMES);
            // Nothing that narrows when a component may be used.
            expect(COMPONENT_GUIDE).not.toMatch(/Most replies need none|WHEN TO USE|is only for|Add a Button only|Do not wrap|Use one only when|must be written as Markdown|one component for one need/i);
            expect(CLOSING_GUIDE).toBe('REMEMBER\nDo not explain components, tags or these instructions to the user.');
            expect(COMPONENT_GUIDE).not.toMatch(/web_search|web_fetch|search the web|do not guess|Never invent/i);
            // The index names every attribute a component takes, so a model that has not read a page still cannot invent one.
            const lineOf = (name) => COMPONENT_GUIDE.split('\n').find((line) => line.startsWith(`- ${name}: `));
            expect(lineOf('FloatingActionButton')).toContain('Attributes: icon*, send, position, ');
            expect(lineOf('ProgressBar')).toContain('Attributes: value*, max, label, showPercentage, ');
            expect(lineOf('Flow')).toContain('Node tags (id*, label, ');
            expect(lineOf('Flow')).toContain('Edge tags (source*, target*, ');
            expect(lineOf('Flow')).toContain('Group tags');
            expect(lineOf('Divider').endsWith('Holds nothing: write it as one tag ending in />.')).toBe(true);
            expect(COMPONENT_GUIDE).toContain('These are all the components there are.');
            expect(COMPONENT_GUIDE).toContain('`<Grid>`, `<Description>`');
            for (const attribute of BOX_ATTRIBUTES) expect(COMPONENT_GUIDE, attribute).toContain(attribute);
            expect(estimateTokens(COMPONENT_GUIDE)).toBeLessThan(1600);
        });

        test('every component has its own page with all of its attributes, and nothing about the others', () => {
            expect(COMPONENT_PAGES.map((page) => page.name).sort()).toEqual([...BLOCK_NAMES].sort());
            const page = (name) => COMPONENT_PAGES.find((one) => one.name === name).text;
            for (const { name, text } of COMPONENT_PAGES) {
                expect(text.startsWith(`COMPONENT: ${name}\n\n`)).toBe(true);
                expect(text).toContain(`<${name}`);
                expect(text).not.toMatch(/Most replies need none|is only for|Use it only|never use it/i);
                // Each example is well formed: every tag that opens also closes.
                for (const example of [...text.matchAll(/```component\n([\s\S]*?)\n```/g)].map((match) => match[1])) {
                    const opened = [...example.matchAll(/<([A-Z]\w*)(?:\s[^<>]*?)?(\/?)>/g)].filter((match) => match[2] !== '/').map((match) => match[1]);
                    const closed = [...example.matchAll(/<\/([A-Z]\w*)>/g)].map((match) => match[1]);
                    expect(opened.sort()).toEqual(closed.sort());
                }
            }
            // Every attribute the chat accepts for a component is on its page, with what it does.
            for (const [name, attributes] of Object.entries(BLOCK_ATTRIBUTES)) {
                for (const attribute of attributes) expect(page(name), `${name}.${attribute}`).toMatch(new RegExp(`\\b${attribute}\\b`));
            }
            for (const part of ['<Row ', '<Column ']) expect(page('Data')).toContain(part);
            for (const part of ['<Node ', '<Edge ']) expect(page('Flow')).toContain(part);
            expect(page('TreeView')).toContain('<Item ');
            expect(page('Select')).toContain('<Option ');
            for (const value of ['solid', 'ghost', 'border-shadow']) expect(page('Button')).toContain(value);
            for (const value of ['grid', 'graph', 'dots', 'aurora', 'between', 'evenly', 'baseline', 'flex-column']) expect(page('Container')).toContain(value);
            expect(page('Badge')).not.toMatch(/Flow|TreeView|backgroundFade/);
            expect(Math.max(...COMPONENT_PAGES.map((one) => estimateTokens(one.text)))).toBeLessThan(2400);
            // A diagram's page says what a box's text may hold and how the person can use it.
            for (const part of ['<Group ', '&#10;', '| Gear | Ratio |', '~~~js', 'What the user can do']) expect(page('Flow')).toContain(part);
        });

        test('is sent with every chat and every model, with or without instructions', async () => {
            const plain = await newChat({ model: 'qwen2.5-coder:7b' });
            const guided = await newChat({ model: 'custom-thing:latest', instructions: [{ name: 'Tone', text: 'Be formal.' }] });
            await send(plain.id, { content: 'one' });
            await send(guided.id, { content: 'two' });
            await send(plain.id, { regenerate: true });

            expect(state.guidesSeen).toBe(3);
            expect(state.chatCalls[0].messages).toEqual([{ role: 'user', content: 'one' }]);
            expect(state.chatCalls[1].messages[0]).toEqual({ role: 'system', content: 'Be formal.' });
        });

        test('is never shown in the chat, the list or an export', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', instructions: [{ name: 'Tone', text: 'Be formal.' }] });
            await send(chat.id, { content: 'hello' });
            const marker = 'draw interface components';
            for (const response of [
                await get(`/chats/${chat.id}`),
                await get('/chats'),
                await get(`/chats/${chat.id}/export?format=markdown`),
                await get(`/chats/${chat.id}/export?format=jsonl`)
            ]) {
                expect(JSON.stringify(response.data)).not.toContain(marker);
            }
            expect((await Chat.findById(chat.id)).system).toBe('');
        });

        test('counts toward what the model can read', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            expect(chat.context.usedTokens).toBeGreaterThanOrEqual(estimateTokens(COMPONENT_GUIDE));
        });
    });

    describe('Instructions made of several articles', () => {
        const tone = { name: 'Tone', text: 'Reply only in formal British English.' };
        const brief = { name: 'brief.md', text: 'The client wants a harbour survey by June.', source: 'fs-one', path: '/docs/brief.md' };

        test('a chat keeps typed text and files as separate articles and sends them all', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', instructions: [tone, brief] });
            expect(chat.instructions).toEqual([
                { ...tone, source: 'ai-base', path: null },
                brief
            ]);

            await send(chat.id, { content: 'hi' });
            expect(state.chatCalls[0].messages[0]).toEqual({
                role: 'system',
                content: '# Tone\n\nReply only in formal British English.\n\n# brief.md\n\nThe client wants a harbour survey by June.'
            });
        });

        test('a single article is sent as it is, without a heading', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', instructions: [tone] });
            await send(chat.id, { content: 'hi' });
            expect(state.chatCalls[0].messages[0].content).toBe('Reply only in formal British English.');
        });

        test('articles can be replaced and cleared', async () => {
            const chat = await newChat({ instructions: [tone, brief] });
            const fewer = await patch(`/chats/${chat.id}`, { instructions: [brief] });
            expect(fewer.data.chat.instructions).toEqual([brief]);
            const none = await patch(`/chats/${chat.id}`, { instructions: [] });
            expect(none.data.chat.instructions).toEqual([]);
            expect(none.data.chat.system).toBe('');
        });

        test('a chat made with the older single instruction shows it as one article', async () => {
            const chat = await newChat({ system: 'Be brief.' });
            expect(chat.instructions).toEqual([{ name: 'Instructions', text: 'Be brief.', source: 'ai-base', path: null }]);
            const changed = await patch(`/chats/${chat.id}`, { instructions: [tone] });
            expect(changed.data.chat.system).toBe(tone.text);
            expect((await Chat.findById(chat.id)).system).toBe('');
        });

        test('articles count toward what the model can read', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', instructions: [{ name: 'Long', text: 'word '.repeat(4000) }] });
            expect(chat.context.usedTokens).toBeGreaterThanOrEqual(5000);
        });

        test('limits: ten articles, each named and not empty', async () => {
            const eleven = Array.from({ length: 11 }, (_, index) => ({ name: `A${index}`, text: 'x' }));
            expect((await post('/chats', { instructions: eleven })).status).toBe(400);
            expect((await post('/chats', { instructions: [{ name: '', text: 'x' }] })).status).toBe(400);
            expect((await post('/chats', { instructions: [{ name: 'A', text: '' }] })).status).toBe(400);
            expect((await post('/chats', { instructions: [{ name: 'A', text: 'a'.repeat(100001) }] })).status).toBe(400);
            expect((await post('/chats', { instructions: [{ name: 'A', text: 'x', source: 'elsewhere' }] })).status).toBe(400);
        });

        test('a new chat from a summary and an export both carry the articles', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b', instructions: [tone, brief] });
            await send(chat.id, { content: 'hello' });
            const exported = (await get(`/chats/${chat.id}/export?format=markdown`)).data.content;
            expect(exported).toContain('# Tone');
            expect(exported).toContain('harbour survey by June');
            const next = (await post(`/chats/${chat.id}/summary`, {})).data.chat;
            expect(next.instructions.map((item) => item.name)).toEqual(['Tone', 'brief.md']);
        });
    });

    describe('Setup guide', () => {
        const put = (body) => client.put('/api/v1/guide', body);

        test('needs sign-in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            expect((await get('/guide')).status).toBe(401);
        });

        test('is on for a first visit, remembers used buttons, and turns itself off when all are used', async () => {
            await Guide.deleteMany({});
            await testStartup.loginAsUser('admin');
            const first = (await get('/guide')).data.guide;
            expect(first).toMatchObject({ on: true, seen: [] });
            expect(first.steps).toEqual(GUIDE_STEPS);
            expect((await Guide.findOne({})).service).toBe('ai-base');

            expect((await post('/guide/seen', { key: 'not.a.button' })).status).toBe(400);
            const once = (await post('/guide/seen', { key: 'chat.send' })).data.guide;
            expect(once).toMatchObject({ on: true, seen: ['chat.send'] });
            expect((await post('/guide/seen', { key: 'chat.send' })).data.guide.seen).toEqual(['chat.send']);
            expect((await get('/guide')).data.guide.seen).toEqual(['chat.send']);

            await testStartup.loginAsUser('user');
            await testStartup.loginAsUser('admin');
            expect((await get('/guide')).data.guide).toMatchObject({ on: true, seen: ['chat.send'] });

            let last = null;
            for (const key of GUIDE_STEPS) last = (await post('/guide/seen', { key })).data.guide;
            expect(last.on).toBe(false);
            expect((await post('/guide/seen', { key: 'chat.new' })).data.guide.on).toBe(false);

            const again = (await put({ on: true })).data.guide;
            expect(again).toMatchObject({ on: true, seen: [] });
            expect((await put({ on: false })).data.guide.on).toBe(false);
            expect((await put({})).status).toBe(400);
            expect(await Guide.countDocuments({})).toBe(1);
        });

        test('each person has their own guide, and a regular user is not asked for manager-only buttons', async () => {
            await Guide.deleteMany({});
            await testStartup.loginAsUser('admin');
            await post('/guide/seen', { key: 'chat.send' });
            await testStartup.loginAsUser('user');
            const mine = (await get('/guide')).data.guide;
            expect(mine).toMatchObject({ on: true, seen: [] });
            expect(mine.steps).toEqual(GUIDE_STEPS.filter((step) => !MANAGER_GUIDE_STEPS.includes(step)));
            let last = null;
            for (const key of mine.steps) last = (await post('/guide/seen', { key })).data.guide;
            expect(last.on).toBe(false);
            expect(await Guide.countDocuments({})).toBe(2);
        });
    });

    describe('Saved instructions', () => {
        const tone = { name: 'Formal tone', text: 'Reply only in formal British English.' };

        test('validation', () => {
            expect(promptSchemas.create.validate(tone).error).toBeUndefined();
            expect(promptSchemas.create.validate({ name: 'x' }).error).toBeDefined();
            expect(promptSchemas.create.validate({ ...tone, name: 'a'.repeat(121) }).error).toBeDefined();
            expect(promptSchemas.update.validate({}).error).toBeDefined();
        });

        test('needs sign-in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            expect((await get('/prompts')).status).toBe(401);
        });

        test('an instruction can be saved, listed, renamed and deleted', async () => {
            const saved = await post('/prompts', tone);
            expect(saved.status).toBe(201);
            expect(saved.data.prompt).toMatchObject({ ...tone, source: 'ai-base', path: null });
            expect((await Prompt.findById(saved.data.prompt.id)).service).toBe('ai-base');

            await post('/prompts', { name: 'A brief', text: 'Survey by June.', source: 'fs-one', path: '/docs/brief.md' });
            const listed = (await get('/prompts')).data.prompts;
            expect(listed.map((prompt) => prompt.name)).toEqual(['A brief', 'Formal tone']);
            expect(listed[0]).toMatchObject({ source: 'fs-one', path: '/docs/brief.md' });

            const renamed = await patch(`/prompts/${saved.data.prompt.id}`, { name: 'Formal' });
            expect(renamed.data.prompt.name).toBe('Formal');
            expect((await remove(`/prompts/${saved.data.prompt.id}`)).status).toBe(200);
            expect((await get('/prompts')).data.prompts).toHaveLength(1);
        });

        test('a saved instruction carries into any number of chats', async () => {
            const saved = (await post('/prompts', tone)).data.prompt;
            const article = { name: saved.name, text: saved.text };
            const first = await newChat({ model: 'qwen2.5-coder:7b', instructions: [article] });
            const second = await newChat({ model: 'qwen2.5-coder:7b', instructions: [article] });
            await send(first.id, { content: 'one' });
            await send(second.id, { content: 'two' });
            expect(state.chatCalls.map((call) => call.messages[0].content)).toEqual([tone.text, tone.text]);

            await remove(`/prompts/${saved.id}`);
            expect((await get(`/chats/${first.id}`)).data.chat.instructions).toHaveLength(1);
        });

        test('saved instructions belong to their owner', async () => {
            const saved = (await post('/prompts', tone)).data.prompt;
            for (const role of ['owner', 'admin']) {
                await testStartup.loginAsUser(role);
                expect((await get('/prompts')).data.prompts).toEqual([]);
                expect((await patch(`/prompts/${saved.id}`, { name: 'Taken' })).status).toBe(404);
                expect((await remove(`/prompts/${saved.id}`)).status).toBe(404);
            }
            expect((await Prompt.findById(saved.id)).name).toBe('Formal tone');
            expect((await get('/prompts/not-an-id')).status).toBeGreaterThanOrEqual(400);
        });

        test('stops at 100 saved instructions', async () => {
            const owner = (await post('/prompts', tone)).data.prompt;
            const stored = await Prompt.findById(owner.id);
            await Prompt.insertMany(Array.from({ length: 99 }, (_, index) => ({ owner: stored.owner, name: `P${index}`, text: 'x' })));
            const response = await post('/prompts', tone);
            expect(response.status).toBe(422);
            expect(response.data.message).toMatch(/100 saved instructions/);
        });
    });

    describe('Loading a model ahead of the first message', () => {
        test('loads the model the next chat will use and keeps it for the set time', async () => {
            await Setting.create({ keepAliveMinutes: 20 });
            const response = await post('/models/load', {});
            expect(response.status).toBe(202);
            expect(response.data).toMatchObject({ model: 'qwen2.5-coder:7b', ready: false, loading: true });
            await wait(30);
            expect(state.unloaded).toEqual([{ model: 'qwen2.5-coder:7b', keep_alive: '20m', num_ctx: 8192 }]);
        });

        test('loads with the same context size a chat will use, so the first message does not load it again', async () => {
            process.env.TAVILY_API_KEY = 'tvly-test';
            try {
                await post('/models/load', {});
                await wait(30);
                expect(state.unloaded[0].num_ctx).toBe(16384);
                const chat = (await post('/chats', { model: 'qwen2.5-coder:7b' })).data.chat;
                await client.post(`/api/v1/chats/${chat.id}/messages`, { content: 'hi' });
                expect(state.chatCalls[state.chatCalls.length - 1].options.num_ctx).toBe(16384);
            } finally {
                process.env.TAVILY_API_KEY = '';
            }
        });

        test('loads a named model', async () => {
            const response = await post('/models/load', { name: 'custom-thing:latest' });
            expect(response.data).toMatchObject({ model: 'custom-thing:latest', loading: true });
        });

        test('does nothing when the model is already in memory', async () => {
            state.running = [{ name: 'qwen2.5-coder:7b', size: 100, size_vram: 100 }];
            const response = await post('/models/load', { name: 'qwen2.5-coder:7b' });
            expect(response.data).toMatchObject({ ready: true, loading: false });
            await wait(30);
            expect(state.unloaded).toEqual([]);
        });

        test('says plainly when Ollama is not running or nothing is installed', async () => {
            state.models = [];
            expect((await post('/models/load', {})).data).toMatchObject({ code: 'NO_MODELS' });
            state.up = false;
            expect((await post('/models/load', {})).status).toBe(503);
        });

        test('needs sign-in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            expect((await post('/models/load', {})).status).toBeGreaterThanOrEqual(401);
        });
    });

    describe('Searching the web', () => {
        const withKey = async (work) => {
            process.env.TAVILY_API_KEY = 'tvly-test';
            try {
                return await work();
            } finally {
                process.env.TAVILY_API_KEY = '';
            }
        };

        beforeAll(() => {
            setSearchFetch(async (url, options = {}) => {
                state.searchUrls.push(String(url));
                const sent = options.body ? JSON.parse(options.body) : null;
                const asked = sent ? sent.query : new URL(url).searchParams.get('q');
                state.searchRequests.push({ method: options.method || 'GET', key: options.headers?.Authorization || null, body: sent });
                if (state.searchFails && !state.searchStatus) throw new Error('ECONNREFUSED');
                if (state.searchFails) return { ok: false, status: state.searchStatus };
                if (state.enginesLimited) return { ok: false, status: 432 };
                state.searches.push(asked);
                const first = state.freshResults ? `https://fresh.example/${state.searches.length}/files/duck.glb` : 'https://models.example/duck';
                const results = Array.from({ length: 8 }, (_, index) => ({ title: index ? `Other ${index}` : 'Duck model', url: index ? `https://other.example/${index}` : first, content: `A duck in glTF format. ${asked}`, engine: 'x' }));
                if (state.junkResults) results.push({ title: 'Free online games', url: 'https://games.example/', content: 'Play free games now.', engine: 'x' }, { title: 'Duck pond model railway', url: 'https://railway.example/duck', content: 'A model of a duck pond.', engine: 'x' });
                if (state.junkResults) results.splice(0, 8);
                return { ok: true, status: 200, json: async () => ({ results }) };
            });
            setRemoteTransport({
                lookup: async (host) => [{ address: host === 'intranet.example' ? '10.0.0.5' : '93.184.216.34', family: 4 }],
                request: async (url) => {
                    state.fetches.push(url.href);
                    if (url.href.endsWith('/files/duck.glb')) return { status: 200, headers: { 'content-type': 'model/gltf-binary' }, stream: Readable.from([Buffer.from('glb')]) };
                    if (url.href.includes('missing')) return { status: 404, headers: {}, stream: Readable.from([]) };
                    if (url.href === 'https://raw.githubusercontent.com/owner/repo/main/data.csv') return { status: 200, headers: { 'content-type': 'text/plain' }, stream: Readable.from([Buffer.from('a,b')]) };
                    if (url.href.endsWith('/icon.png')) return { status: 200, headers: { 'content-type': 'image/png', 'content-length': '900' }, stream: Readable.from([Buffer.from('png')]) };
                    if (url.host === 'other.example') return { status: 200, headers: { 'content-type': 'text/html' }, stream: Readable.from([Buffer.from('<html><body><a href="https://models.example/files/missing.glb">a</a><a href="https://models.example/files/page.glb">b</a><img src="/icon.png"><a href="https://github.com/owner/repo/blob/main/data.csv">c</a><script>window.data = {"file":"https:\\/\\/models.example\\/hidden\\/files\\/duck.glb"}</script></body></html>')]) };
                    return { status: 200, headers: { 'content-type': 'text/html; charset=utf-8' }, stream: Readable.from([Buffer.from("<html><head><title>Duck &amp; friends</title><style>p{color:red}</style></head><body><p>Download the duck below.</p><a href=\"/files/duck.glb\">file</a> <a href=\"https://models.example/about\">about</a><script>evil()</script></body></html>")]) };
                }
            });
        });

        afterAll(() => {
            setSearchFetch(null);
            setRemoteTransport(null);
        });
        const search = ['web_search', { query: 'duck 3d model' }];
        const fetchPage = ['web_fetch', { url: 'https://models.example/duck' }];

        test('a long chat folds its older messages into a memory, sends the memory and the recent messages, and drops it when an edit reaches back', async () => {
            const usual = state.contextLength;
            state.contextLength = 4096;
            state.memoryCalls = [];
            state.memoryReply = '- The user is planning a duck pond.\n- The pond is 3 m across.';
            try {
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const say = (n) => `${n} ${'word '.repeat(120)}`;
                for (const n of [1, 2, 3]) await send(chat.id, { content: say(n) });
                // Under the threshold, or with nothing older than the last six messages, nothing is folded.
                expect(state.memoryCalls).toHaveLength(0);
                await send(chat.id, { content: say(4) });
                let saved = null;
                for (let tries = 0; tries < 100; tries += 1) {
                    saved = await Chat.findById(chat.id);
                    if (saved.memoryThrough > 0) break;
                    await wait(20);
                }
                // The first exchange is folded, by the chat's own model, with the last six messages kept word for word.
                expect(saved.memoryThrough).toBe(2);
                expect(saved.memory).toBe(state.memoryReply);
                expect(state.memoryCalls).toHaveLength(1);
                expect(state.memoryCalls[0]).toMatchObject({ model: 'qwen2.5-coder:7b', stream: false });
                const asked = state.memoryCalls[0].messages[0].content;
                expect(asked).toMatch(/^Rewrite the memory of this conversation/);
                expect(asked).toContain(`The messages to add:\nUser: ${say(1).trim()}`);
                expect(asked).not.toContain('User: 2 ');

                // The next message is sent with the memory, and only the messages after it.
                state.chatCalls.length = 0;
                await send(chat.id, { content: say(5) });
                const sent = state.chatCalls[0].messages;
                expect(sent[0]).toMatchObject({ role: 'system', content: expect.stringContaining(`Memory of the earlier part of this conversation (those messages are not repeated below):\n${state.memoryReply}`) });
                expect(sent.filter((message) => message.role === 'user').map((message) => message.content.split(' ')[0])).toEqual(['2', '3', '4', '5']);
                const shownChat = (await get(`/chats/${chat.id}`)).data.chat;
                expect(shownChat.memoryThrough).toBeGreaterThanOrEqual(2);
                expect(shownChat.memory).toBe(state.memoryReply);
                expect((await get(`/chats/${chat.id}`)).data.chat.messages).toHaveLength(10);

                // Editing a message the memory covers drops the memory.
                await send(chat.id, { content: 'new first', editIndex: 0 });
                const edited = await Chat.findById(chat.id);
                expect(edited.memoryThrough).toBe(0);
                expect(edited.memory).toBe('');
            } finally {
                state.contextLength = usual;
                state.memoryReply = null;
            }
        });

        test('within a reply, a tool result the model has moved past is put away, and the same call brings it back whole', async () => {
            await withKey(async () => {
                // The search repeats the query in each result, so a long query makes a result long enough to put away.
                const ask = (n) => `duck model ${n} in glTF with a rig and textures`;
                state.toolPlan = [...[0, 1, 2, 3].map((n) => [['web_search', { query: ask(n) }]]), [['web_search', { query: ask(0) }]]];
                state.chatReply = 'Here are the ducks.';
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'find duck models' });
                expect(list.at(-1).type).toBe('done');
                // The repeated search is not run again.
                expect(state.searches).toEqual([0, 1, 2, 3].map(ask));
                const results = state.chatCalls.at(-1).messages.filter((message) => message.role === 'tool');
                expect(results).toHaveLength(5);
                // Three steps back or more: put away, saying how to get it back. The last two steps stay whole.
                expect(results[0].content).toMatch(/^\[Put away to save room: what web_search returned for \{"query":"duck model 0 in glTF with a rig and textures"\}, [\d,]+ characters, read \d steps ago\. Call web_search again with exactly the same input to see it again, or keep what you need with save_note\.\]$/);
                expect(results[1].content).toMatch(/^\[Put away to save room: what web_search returned for \{"query":"duck model 1 in glTF/);
                for (const kept of results.slice(2, 4)) expect(kept.content).not.toMatch(/^\[Put away/);
                // Asked for again, the put-away result comes back whole, not as a reminder.
                expect(results[4].content).not.toMatch(/^\[Put away|^You already called/);
                expect(results[4].content).toContain('duck model 0');
                expect(results[4].content.length).toBeGreaterThan(600);
            });
        });

        test('a model keeps notes with save_note: they go with every later message, a key replaces, and empty text removes', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.toolPlan = [[['save_note', { key: 'battery', text: '36 V, 14 Ah' }], ['save_note', { key: 'motor', text: '250 W hub' }]]];
            const { events: first } = await send(chat.id, { content: 'remember the battery and motor' });
            expect(first.filter((event) => event.type === 'tool').map((event) => event.tool)).toEqual(['Note', 'Note']);
            expect((await get(`/chats/${chat.id}`)).data.chat.notes.map(({ key, text }) => ({ key, text }))).toEqual([{ key: 'battery', text: '36 V, 14 Ah' }, { key: 'motor', text: '250 W hub' }]);

            state.chatCalls.length = 0;
            state.toolPlan = [[['save_note', { key: 'battery', text: '48 V, 14 Ah' }], ['save_note', { key: 'motor', text: '' }]]];
            await send(chat.id, { content: 'it is 48 V, and forget the motor' });
            // The notes as they stood go with the message; the changes are kept for the next.
            expect(state.chatCalls[0].messages[0].content).toContain('Notes saved in this chat with save_note:\n- battery: 36 V, 14 Ah\n- motor: 250 W hub');
            state.chatCalls.length = 0;
            await send(chat.id, { content: 'thanks' });
            expect(state.chatCalls[0].messages[0].content).toContain('Notes saved in this chat with save_note:\n- battery: 48 V, 14 Ah');
            expect(state.chatCalls[0].messages[0].content).not.toContain('motor');
        });

        test('is off without a search address: no tools are offered, the page is told so, and the model gets no extra instruction', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            await send(chat.id, { content: 'find a duck model' });
            expect(state.chatCalls[0].tools).toBeUndefined();
            expect(COMPONENT_GUIDE).not.toMatch(/web_search/);
            expect(state.chatCalls[0].options.num_ctx).toBe(8192);
            expect((await get('/models')).data.webSearch).toBe(false);
        });

        test('with a search address, both tools are offered with more room to read, and nothing of Mollama\'s own tells the model how to use them', async () => {
            await withKey(async () => {
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                await send(chat.id, { content: 'find a duck model' });
                const request = state.chatCalls[0];
                expect(request.tools.map((tool) => tool.function.name)).toEqual(['web_search', 'web_fetch']);
                // Only the user's message is left once the default skill and the date are taken off: no search rules.
                expect(request.messages.map((message) => message.role)).toEqual(['user']);
                expect(request.options.num_ctx).toBe(16384);

                // The guidance for searching is a skill the chat may turn on. A model that can use tools is handed its name
                // and description, not its text, and a call to web_search is never mistaken for opening it.
                state.chatCalls.length = 0;
                state.toolPlan = [[search]];
                const guided = await newChat({ model: 'qwen2.5-coder:7b', skills: ['search-web-search'] });
                const { events: steps } = await send(guided.id, { content: 'find a duck model' });
                expect(state.chatCalls[0].messages.map((message) => message.role)).toEqual(['user']);
                const opener = state.chatCalls[0].tools.find((tool) => tool.function.name === 'open_skill');
                expect(opener.function.description).toMatch(/\n- Web Search: When to search the web/);
                expect(opener.function.parameters.properties.skill.enum).toEqual(['Web Search']);
                expect(steps.filter((event) => event.type === 'tool').map((event) => event.tool)).toEqual(['web_search']);
                expect((await get('/models')).data.webSearch).toBe(true);
            });
        });

        test('with web search on, its page reader stands in for the built-in one, so only one of the two is offered', async () => {
            await withKey(async () => {
                state.toolsSeen = [];
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                await send(chat.id, { content: 'find a duck model' });
                expect(state.toolsSeen[0].map((tool) => tool.function.name)).toEqual(['web_search', 'web_fetch', 'open_component', 'open_skill_file', 'open_skill', 'find_skill', 'ask_user', 'save_note', 'list_my_files', 'read_my_file', 'save_to_files', 'date_time', 'calculator']);

                // Switched off for the chat, the search tools go and the built-in reader is back.
                state.toolsSeen = [];
                const quiet = await newChat({ model: 'qwen2.5-coder:7b', webSearch: false });
                await send(quiet.id, { content: 'find a duck model' });
                expect(state.toolsSeen[0].map((tool) => tool.function.name)).toEqual(['open_component', 'open_skill_file', 'open_skill', 'find_skill', 'ask_user', 'save_note', 'list_my_files', 'read_my_file', 'save_to_files', 'date_time', 'calculator', 'read_web_page']);
            });
        });

        test('a model that cannot use tools is not offered them', async () => {
            await withKey(async () => {
                state.noTools = true;
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                await send(chat.id, { content: 'find a duck model' });
                expect(state.chatCalls[0].tools).toBeUndefined();
            });
        });

        test('searches, reads a page, then answers from what it found', async () => {
            await withKey(async () => {
                state.toolPlan = [[search], [fetchPage]];
                state.chatReply = 'Here is the duck.';
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'find a duck model' });

                expect(list.filter((event) => event.type === 'tool')).toEqual([
                    { type: 'tool', tool: 'web_search', input: 'duck 3d model' },
                    { type: 'tool', tool: 'web_fetch', input: 'https://models.example/duck' }
                ]);
                expect(state.searches).toEqual(['duck 3d model']);
                expect(state.searchUrls).toEqual(['https://api.tavily.com/search']);
                expect(state.fetches).toEqual(['https://models.example/duck']);
                expect(state.chatCalls).toHaveLength(3);

                const last = state.chatCalls[2].messages;
                const results = last.filter((message) => message.role === 'tool');
                expect(results.map((message) => message.tool_name)).toEqual(['web_search', 'web_fetch']);
                expect(results[0].content).toContain('https://models.example/duck');
                expect(JSON.parse(results[0].content)).toHaveLength(5);
                expect(JSON.parse(results[1].content)).toEqual({ title: 'Duck & friends', content: 'Download the duck below.\nfile about', links: ['https://models.example/files/duck.glb', 'https://models.example/about'] });
                expect(last.filter((message) => message.role === 'assistant' && message.tool_calls)).toHaveLength(2);

                const done = list[list.length - 1];
                expect(done).toMatchObject({ type: 'done', reply: { content: 'Here is the duck.' } });
                expect(done.reply.steps).toEqual([
                    { tool: 'web_search', input: 'duck 3d model', failed: null, answer: expect.any(String) },
                    { tool: 'web_fetch', input: 'https://models.example/duck', failed: null, answer: expect.any(String) }
                ]);
                expect((await get(`/chats/${chat.id}`)).data.chat.messages[1].steps).toHaveLength(2);
            });
        });

        test('accepts tool arguments written as text, and refuses to read something that is not a web address', async () => {
            await withKey(async () => {
                state.toolPlan = [[['web_search', '{"query":"duck"}'], ['web_fetch', { url: 'file:///etc/passwd' }]]];
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                await send(chat.id, { content: 'find a duck model' });
                expect(state.searches).toEqual(['duck']);
                expect(state.fetches).toEqual([]);
                expect(state.chatCalls[1].messages.filter((message) => message.role === 'tool')[1].content).toMatch(/https/);
            });
        });

        test('a failed search is told to the model, and the reply still finishes', async () => {
            await withKey(async () => {
                state.searchFails = true;
                state.toolPlan = [[search]];
                state.chatReply = 'I could not look that up.';
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'find a duck model' });
                expect(state.chatCalls[1].messages.find((message) => message.role === 'tool').content).toBe('Error: The search service could not be reached. (ECONNREFUSED)');
                expect(list[list.length - 1]).toMatchObject({ type: 'done', reply: { content: 'I could not look that up.', error: null } });
                expect(list.find((event) => event.type === 'toolFailed').failed).toBe('unreachable');
            });
        });

        test.each([[403, 'refused', 'The search service refused the request.'], [429, 'limited', 'The search service is limiting requests right now.']])('a search answered with %i is told to the model plainly and kept on the reply', async (status, failed, said) => {
            await withKey(async () => {
                state.searchFails = true;
                state.searchStatus = status;
                state.toolPlan = [[search]];
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'find a duck model' });
                expect(state.chatCalls[1].messages.find((message) => message.role === 'tool').content).toBe(`Error: ${said} (status ${status})`);
                expect(list.find((event) => event.type === 'toolFailed').failed).toBe(failed);
                expect(list[list.length - 1].reply.steps[0].failed).toBe(failed);
                expect((await get(`/chats/${chat.id}`)).data.chat.messages[1].steps[0].failed).toBe(failed);
            });
        });

        test('tool calls a model writes as text are run as calls and kept out of the reply', async () => {
            await withKey(async () => {
                state.toolsAsText = true;
                state.toolPlan = [[search, ['web_search', { query: 'second' }], ['web_search', { query: 'third' }], ['web_search', { query: 'fourth' }], ['web_search', { query: 'fifth' }]], [fetchPage]];
                state.chatReply = 'Here is the duck.';
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'find a duck model' });
                expect(state.searches).toEqual(['duck 3d model', 'second', 'third', 'fourth', 'fifth']);
                expect(state.fetches).toEqual(['https://models.example/duck']);
                expect(list.filter((event) => event.type === 'replace')).toEqual([{ type: 'replace', text: '' }, { type: 'replace', text: '' }]);
                const done = list[list.length - 1];
                expect(done.reply.content).toBe('Here is the duck.');
                expect(done.reply.steps).toHaveLength(6);
                const second = state.chatCalls[1].messages;
                expect(second.find((message) => message.tool_calls).tool_calls).toHaveLength(5);
                expect(second.filter((message) => message.role === 'tool')).toHaveLength(5);
            });
        });

        test('tool calls written at the end of a reply are found, and anything else is left alone', () => {
            const call = '{"name": "web_search", "arguments": {"query": "a {b} \\"c\\""}}';
            const found = { function: { name: 'web_search', arguments: { query: 'a {b} "c"' } } };
            expect(writtenToolCalls(call)).toEqual({ calls: [found], text: '' });
            expect(writtenToolCalls(`<tool_call>\n${call}\n</tool_call>`)).toEqual({ calls: [found], text: '' });
            expect(writtenToolCalls(`\`\`\`json\n[${call},\n{"name":"web_fetch","parameters":{"url":"https://a.example"}}]\n\`\`\``).calls).toHaveLength(2);
            expect(writtenToolCalls(`I will search first.\n\n${call}\n${call}`)).toEqual({ calls: [found, found], text: 'I will search first.' });
            expect(writtenToolCalls(`${call}\nThen I will answer.`).calls).toEqual([]);
            expect(writtenToolCalls('{"name": "delete_everything", "arguments": {}}').calls).toEqual([]);
            expect(writtenToolCalls('{"name": "web_search", "arguments": {"query": "cut off').calls).toEqual([]);
            expect(writtenToolCalls('Here is the duck.')).toEqual({ calls: [], text: 'Here is the duck.' });
        });

        test('will not read a page on a private network', async () => {
            await withKey(async () => {
                state.toolPlan = [[['web_fetch', { url: 'https://intranet.example/admin' }]]];
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'read the intranet' });
                expect(state.fetches).toEqual([]);
                expect(state.chatCalls[1].messages.find((message) => message.role === 'tool').content).toBe('Error: This address cannot be loaded.');
                expect(list.find((event) => event.type === 'toolFailed').failed).toBe('blocked');
            });
        });

        test('a model opens only the component page it needs, the page is gone after that reply, and nothing nudges it to draw', async () => {
            const page = (name) => COMPONENT_PAGES.find((one) => one.name === name).text;
            const pageText = (request) => COMPONENT_PAGES.filter((one) => JSON.stringify(request.messages).includes(JSON.stringify(one.text).slice(1, -1))).map((one) => one.name);
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.toolsSeen = [];
            state.toolPlan = [[['open_component', { component: 'Flow, badge' }]]];
            state.chatReply = 'Here is the diagram.';
            const { events: list } = await send(chat.id, { content: 'draw the steps' });

            // Only the index is sent; no page is in the request until the model asks for one.
            expect(state.guidesSeen).toBeGreaterThan(0);
            expect(state.chatCalls[0].messages.map((message) => message.role)).toEqual(['user']);
            expect(pageText(state.chatCalls[0])).toEqual([]);
            const opener = state.toolsSeen[0].find((tool) => tool.function.name === 'open_component');
            expect(opener.function.description).toMatch(/every attribute it takes, every value, and what each does on screen/);
            // A model is told the function is for reading: some took a page for the drawing itself and said they could not draw.
            expect(opener.function.description).toMatch(/only returns the page and draws nothing: a component is drawn by writing its tags in your reply, inside a ```component fence\.$/);
            // The two it asked for come back together, and only those two.
            expect(state.chatCalls[1].messages.at(-1)).toMatchObject({ role: 'tool', content: `${page('Flow')}\n\n${page('Badge')}` });
            expect(pageText(state.chatCalls[1])).toEqual(['Badge', 'Flow']);
            expect(list.find((event) => event.type === 'tool')).toMatchObject({ tool: 'Component', input: 'Flow, Badge' });

            // The next message starts clean: the pages are not carried along, and the function is still there.
            state.chatCalls.length = 0;
            state.toolsSeen = [];
            state.toolPlan = [[['Card', {}], ['open_component', { component: 'Chip' }]]];
            const { events: more } = await send(chat.id, { content: 'and a panel' });
            expect(pageText(state.chatCalls[0])).toEqual([]);
            expect(state.chatCalls[0].messages.some((message) => message.role === 'system')).toBe(false);
            expect(state.toolsSeen[0].some((tool) => tool.function.name === 'open_component')).toBe(true);
            // A component called by its own name is opened all the same; a name that is no component is answered plainly.
            const answers = state.chatCalls[1].messages.filter((message) => message.role === 'tool');
            expect(answers[0].content).toBe(page('Card'));
            expect(answers[1].content).toMatch(/^There is no component by that name\. The components are: Audio, Badge, /);
            expect(more.at(-1).reply.steps.map(({ tool, input, failed }) => ({ tool, input, failed: failed || null }))).toEqual([{ tool: 'Component', input: 'Card', failed: null }, { tool: 'Component', input: 'Chip', failed: 'failed' }]);

            // However many replies go by without a component, nothing is added to suggest one.
            const quiet = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatCalls.length = 0;
            for (let turn = 0; turn < 6; turn++) await send(quiet.id, { content: `question ${turn}` });
            expect(state.chatCalls.every((call) => call.messages[0].role === 'user')).toBe(true);

            // A model that can use tools is never sent the compact page of values: it opens the page it needs.
            expect(state.valuesSeen).toBe(0);

            // A model that cannot use tools cannot open a page, so after the index it is sent one compact page of the
            // values each attribute takes, built from the pages' own tables; never the pages themselves.
            expect(COMPONENT_VALUES_PAGE.startsWith('COMPONENT VALUES\n')).toBe(true);
            expect(COMPONENT_VALUES_PAGE).toMatch(/\n- Badge: color = default, primary, secondary, tertiary, success, warning, error/);
            expect(COMPONENT_VALUES_PAGE.length).toBeLessThan(6000);
            state.noTools = true;
            state.chatCalls.length = 0;
            const before = state.guidesSeen;
            const plain = await newChat({ model: 'qwen2.5-coder:7b' });
            await send(plain.id, { content: 'hello' });
            expect(state.guidesSeen).toBe(before + 1);
            expect(state.valuesSeen).toBe(1);
            expect(pageText(state.chatCalls[0])).toEqual([]);
            expect(state.chatCalls[0].messages.map((message) => message.role)).toEqual(['user']);
            state.noTools = false;
            state.toolsSeen = [];
            const bare = await newChat({ model: 'qwen2.5-coder:7b', skills: [] });
            await send(bare.id, { content: 'hello' });
            expect(state.toolsSeen[0].some((tool) => tool.function.name === 'open_component')).toBe(false);
        });

        test('the skill library is whole: every skill has a name, a description, a known project and a licence that allows it', async () => {
            const { skills, sources, maxSkills } = (await get('/skills')).data;
            expect(skills.length).toBeGreaterThan(250);
            expect(new Set(skills.map((skill) => skill.id)).size).toBe(skills.length);
            for (const skill of SKILLS) {
                expect(skill.id).toMatch(/^[a-z0-9-]{1,60}$/);
                expect(skill.name.length).toBeGreaterThan(1);
                expect(SKILL_SOURCES[skill.source]).toBeTruthy();
                if (skill.source === 'mollama') continue;
                expect(skill.description.length).toBeGreaterThan(10);
                expect(skill.credit).toMatch(/\((MIT|Apache-2\.0|MIT, Apache-2\.0)\)$/);
                expect(skill.text.startsWith(`SKILL: ${skill.name}`)).toBe(true);
            }
            // Mollama's own skills: the one that draws its viewers and the map of what each skill is for.
            expect(skills.filter((skill) => skill.source === 'mollama').map((skill) => skill.id)).toEqual(['components', 'skill-map']);
            expect(sources.map((source) => source.id)).toEqual(['mollama', 'anthropic', 'engineering', 'marketing', 'medical', 'dex', 'context7', 'search', 'last30days', 'anatomy', 'hairline', 'taste', 'richmd', 'explainer', 'cadsmith']);
            expect(sources.reduce((sum, source) => sum + source.count, 0)).toBe(skills.length);
            expect(skills.every((skill) => skill.tokens > 0)).toBe(true);

            // Nothing of Mollama's own is put in front of a library skill: its text is its name and the author's page.
            for (const skill of SKILLS.filter((one) => one.source !== 'mollama')) {
                expect(skill.text.startsWith(`SKILL: ${skill.name}\n\n`)).toBe(true);
                expect(skill.text).not.toMatch(/written for an agent that can run commands|Never pretend to have run a command/);
            }

            // What a skill still lacks is said only where it is true: a line that installs the skill is not a command to
            // run, a skill whose files and functions are here lacks nothing, and one without them says so.
            const lacks = (id) => skills.find((skill) => skill.id === id).needs;
            expect(lacks('medical-board-exam-prep')).toEqual([]);
            expect(lacks('anthropic-claude-api')).not.toContain('commands');
            expect(lacks('hairline-hairline-create')).toEqual([]);
            expect(lacks('anatomy-anatomy')).toEqual(['commands']);
            expect(lacks('anthropic-docx')).toEqual(['commands']);
            expect(skills.filter((skill) => skill.source === 'medical' && skill.needs.includes('commands'))).toEqual([]);

            // Every skill says what running it on a person's own computer takes, read from its own page: where it comes
            // from, its author's install line, the programs its commands call, and whether it runs commands at all.
            const setupOf = async (id) => (await get(`/skills/${id}`)).data.skill.setup;
            expect(await setupOf('anthropic-xlsx')).toMatchObject({ repos: ['anthropics/skills'], runs: true, programs: expect.arrayContaining(['python', 'libreoffice']) });
            expect((await setupOf('anthropic-pdf')).libraries).toEqual(expect.arrayContaining(['pdf2image']));
            const prep = await setupOf('medical-board-exam-prep');
            expect(prep).toMatchObject({ runs: false, install: ['npx skills add Open-Medica/open-medical-skills --skill board-exam-prep'], libraries: [] });
            // The install line needs Node; it is not counted as a command the skill runs.
            expect(prep.programs).toEqual(['node']);
            for (const skill of SKILLS.filter((one) => one.source !== 'mollama')) expect((await setupOf(skill.id)).repos.length).toBeGreaterThan(0);

            // A chat holds only a few skills at once.
            expect(maxSkills).toBe(MAX_CHAT_SKILLS);
            const many = SKILLS.filter((skill) => skill.source === 'engineering').slice(0, MAX_CHAT_SKILLS + 4).map((skill) => skill.id);
            const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: many });
            expect(chat.skills).toHaveLength(MAX_CHAT_SKILLS);
        });

        test('a person adds a skill by its address, reads it, uses it in a chat and removes it; nobody else can', async () => {
            const any = { validateStatus: () => true };
            const asked = [];
            const file = (text) => ({ status: 200, headers: { 'content-type': 'text/plain' }, stream: Readable.from([Buffer.from(text)]) });
            const pages = {
                'https://raw.githubusercontent.com/someone/skills/main/skills/plain-talk/SKILL.md': '---\nname: plain-talk\ndescription: >\n  Answers in short plain sentences,\n  one idea at a time.\n---\n\n# Plain talk\n\nUse short sentences. One idea in each.',
                'https://raw.githubusercontent.com/someone/skills/main/web/SKILL.md': '<!DOCTYPE html><html><body>Not a skill</body></html>',
                'https://raw.githubusercontent.com/someone/skills/main/big/SKILL.md': `---\nname: big\ndescription: Too big\n---\n${'x'.repeat(130 * 1024)}`
            };
            const transportBefore = setRemoteTransport({
                lookup: async (host) => [{ address: host === 'intranet.example' ? '10.0.0.5' : '93.184.216.34', family: 4 }],
                request: async (url) => {
                    asked.push(url.href);
                    return pages[url.href] ? file(pages[url.href]) : { status: 404, headers: {}, stream: Readable.from([]) };
                }
            });
            try {
                await testStartup.loginAsUser('user');
                const address = 'https://github.com/someone/skills/tree/main/skills/plain-talk';
                const added = await client.post('/api/v1/skills', { address }, any);
                expect(added.status).toBe(201);
                expect(asked).toEqual(['https://raw.githubusercontent.com/someone/skills/main/skills/plain-talk/SKILL.md']);
                const skill = added.data.skill;
                expect(skill).toMatchObject({ name: 'Plain talk', description: 'Answers in short plain sentences, one idea at a time.', source: 'mine', credit: address, needs: [] });
                expect(skill.id).toMatch(/^mine-[0-9a-f]{24}$/);
                expect(skill.text).toBeUndefined();

                const listed = (await get('/skills')).data;
                expect(listed.skills.find((one) => one.id === skill.id)).toMatchObject({ name: 'Plain talk', source: 'mine' });
                expect(listed.sources.find((source) => source.id === 'mine')).toMatchObject({ name: 'My skills', count: 1 });
                const read = (await get(`/skills/${skill.id}`)).data.skill;
                expect(read.text).toBe('SKILL: Plain talk\n\n# Plain talk\n\nUse short sentences. One idea in each.');

                // Adding the same address again reads it afresh; it does not make a second copy.
                pages['https://raw.githubusercontent.com/someone/skills/main/skills/plain-talk/SKILL.md'] += '\nNo filler words.';
                const again = await client.post('/api/v1/skills', { address }, any);
                expect(again.status).toBe(200);
                expect(again.data.skill.id).toBe(skill.id);
                expect((await get(`/skills/${skill.id}`)).data.skill.text).toMatch(/No filler words\.$/);

                const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: [skill.id] });
                expect(chat.skills).toEqual([skill.id]);
                await send(chat.id, { content: 'hello' });
                expect(state.chatCalls.at(-1).tools.find((tool) => tool.function.name === 'open_skill').function.description).toContain('- Plain talk: Answers in short plain sentences, one idea at a time.');
                state.noTools = true;
                const plainChat = await newChat({ model: 'qwen2.5-coder:7b', skills: [skill.id] });
                await send(plainChat.id, { content: 'hello' });
                state.noTools = false;
                expect(state.chatCalls.at(-1).messages[0].content).toContain('SKILL: Plain talk\n\n# Plain talk');

                // What is not a skill file, too large, missing, or not a public secure address is refused.
                expect((await client.post('/api/v1/skills', { address: 'https://github.com/someone/skills/tree/main/web' }, any)).status).toBe(422);
                expect((await client.post('/api/v1/skills', { address: 'https://github.com/someone/skills/tree/main/big' }, any)).status).toBe(422);
                expect((await client.post('/api/v1/skills', { address: 'https://github.com/someone/skills/tree/main/gone' }, any)).status).toBe(404);
                expect((await client.post('/api/v1/skills', { address: 'https://intranet.example/SKILL.md' }, any)).status).toBe(400);
                expect((await client.post('/api/v1/skills', { address: 'http://example.com/SKILL.md' }, any)).status).toBe(400);
                expect((await get('/skills')).data.skills.filter((one) => one.source === 'mine')).toHaveLength(1);

                // It belongs to the person who added it.
                await testStartup.loginAsUser('creator');
                expect((await get('/skills')).data.skills.some((one) => one.id === skill.id)).toBe(false);
                expect((await client.get(`/api/v1/skills/${skill.id}`, any)).status).toBe(404);
                expect((await client.delete(`/api/v1/skills/${skill.id}`, any)).status).toBe(404);
                const theirs = await newChat({ model: 'qwen2.5-coder:7b', skills: [skill.id] });
                expect(theirs.skills).toEqual([]);

                await testStartup.loginAsUser('user');
                expect((await client.delete(`/api/v1/skills/${skill.id}`, any)).status).toBe(200);
                expect((await get('/skills')).data.skills.some((one) => one.id === skill.id)).toBe(false);
                state.chatCalls.length = 0;
                await send(chat.id, { content: 'hello again' });
                expect(state.chatCalls.at(-1).messages.some((message) => message.role === 'system' && message.content.includes('Plain talk'))).toBe(false);
            } finally {
                setRemoteTransport(transportBefore);
                await Skill.deleteMany({});
            }
        });

        test('nothing of Mollama\'s own is sent or listed, and the skills suggested for small models are short and need nothing extra', async () => {
            const listed = (await get('/skills')).data;
            const { skills } = listed;
            expect(listed.always).toBeUndefined();
            for (const gone of ['base-assistant', 'base-tools', 'base-search-plan', 'base-search-results']) {
                expect((await client.get(`/api/v1/skills/${gone}`, { validateStatus: () => true })).status).toBe(404);
            }
            for (const gone of ['assistant.md', 'search.md', 'tools.md']) {
                expect(fs.existsSync(new URL(`../../server/instructions/${gone}`, import.meta.url))).toBe(false);
            }
            // A chat with every skill off is sent nothing but the date; with no instructions of its own it has no other system text.
            state.toolsSeen = [];
            const bare = await newChat({ model: 'qwen2.5-coder:7b', skills: [], tools: [] });
            await send(bare.id, { content: 'hello' });
            expect(state.chatCalls.at(-1).messages.map((message) => message.role)).toEqual(['user']);
            expect(state.toolsSeen.at(-1).map((tool) => tool.function.name)).toEqual(['find_skill', 'ask_user', 'save_note', 'list_my_files', 'read_my_file', 'save_to_files', 'date_time', 'calculator', 'read_web_page']);

            const suggested = skills.filter((skill) => skill.suggested);
            expect(suggested.map((skill) => skill.id).sort()).toEqual([...SUGGESTED_SKILLS].sort());
            for (const skill of suggested) {
                expect(skill.needs).toEqual([]);
                expect(skill.tokens).toBeLessThan(1600);
            }
            const whole = (await get('/skills/engineering-grilling')).data.skill;
            expect(whole.text.startsWith('SKILL: Grilling')).toBe(true);
            expect((await client.get('/api/v1/skills/no-such-skill', { validateStatus: () => true })).status).toBe(404);
        });

        test('skills are listed, and a chat sends only the ones it has turned on', async () => {
            expect((await get('/skills')).data.webSearch).toBe(false);
            const listed = (await get('/skills')).data.skills;
            expect(listed.map((skill) => skill.id)).toEqual(SKILLS.map((skill) => skill.id));
            expect(listed.find((skill) => skill.id === 'components')).toMatchObject({ default: true });
            expect(listed.filter((skill) => skill.default).map((skill) => [skill.id, skill.default])).toEqual([['components', true], ['skill-map', true], ['richmd-rich-md-first', 'tools'], ['explainer-visual-explainer', 'tools']]);
            expect(listed.find((skill) => skill.id === 'marketing-copywriting')).toMatchObject({ name: 'Copywriting', source: 'marketing', credit: 'coreyhaines31/marketingskills (MIT)', default: false });
            expect(listed[0].text).toBeUndefined();

            // A model that cannot use tools is sent each skill in full.
            state.noTools = true;
            const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: ['marketing-copywriting', 'not-a-skill'] });
            expect(chat.skills).toEqual(['marketing-copywriting']);
            await send(chat.id, { content: 'hello' });
            state.noTools = false;
            expect(state.chatCalls[0].tools).toBeUndefined();
            const system = state.chatCalls[0].messages[0].content;
            expect(system).toContain(SKILLS.find((skill) => skill.id === 'marketing-copywriting').text);
            expect(system).toContain('SKILL: Copywriting');
            expect(system).not.toContain('HOW TO DRAW');

            const none = (await client.patch(`/api/v1/chats/${chat.id}`, { skills: [] })).data.chat;
            expect(none.skills).toEqual([]);
            const back = (await client.patch(`/api/v1/chats/${chat.id}`, { skills: null })).data.chat;
            expect(back.skills).toEqual(['components', 'skill-map', 'richmd-rich-md-first', 'explainer-visual-explainer']);
            expect((await client.patch(`/api/v1/chats/${chat.id}`, { skills: ['Bad Id'] })).status).toBe(400);
        });

        test('a model that can use tools is handed each skill by name, opens one when it fits, and keeps what it opened', async () => {
            const copy = SKILLS.find((skill) => skill.id === 'marketing-copywriting');
            const grill = SKILLS.find((skill) => skill.id === 'engineering-grilling');
            const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: ['components', 'marketing-copywriting', 'engineering-grilling'] });
            state.toolPlan = [[['open_skill', { skill: 'copywriting' }]]];
            state.chatReply = 'Here is the headline.';
            const { events: list } = await send(chat.id, { content: 'write a headline' });

            // Mollama's own skill is sent in full; the others only by name and description, inside one function.
            const first = state.chatCalls[0];
            expect(state.guidesSeen).toBeGreaterThan(0);
            expect(first.messages.map((message) => message.role)).toEqual(['user']);
            const opener = (request) => request.tools.find((tool) => tool.function.name === 'open_skill');
            expect(opener(first).function.description).toContain(`- Copywriting: ${copy.description}`);
            // A model is told whose the instructions are: some wrote a skill's own questions back to the user as the answer.
            expect(opener(first).function.description).toMatch(/weigh every skill on it against the request and open each one that bears on it, not only the first that fits/);
            expect(opener(first).function.description).toMatch(/^Open one of this chat's skills and get its full instructions\. .*The instructions are written for you and the user does not see them: do what they say in your reply, and do not repeat them back\.\n- /);
            expect(opener(first).function.description).toContain(`- Grilling: ${grill.description}`);
            expect([...opener(first).function.parameters.properties.skill.enum].sort()).toEqual(['Copywriting', 'Grilling']);
            // Room is kept for the largest skill that may be opened.
            expect(first.options.num_ctx).toBeGreaterThanOrEqual(8192);

            // The skill's own page comes back as the answer, and the chat shows the step.
            // It ends with the skills it goes with, those on in the chat apart from those in the library.
            const opened = state.chatCalls[1].messages.at(-1);
            expect(opened.role).toBe('tool');
            expect(opened.content.startsWith(copy.text)).toBe(true);
            expect(opened.content.slice(copy.text.length)).toMatch(/^\n\nGOES WITH\n(On in this chat, open with open_skill when the request needs it: .+\.\n)?In the library, not on: .+\. When one would help, tell the user its name; they turn it on under Skills\.$/);
            expect(list.find((event) => event.type === 'tool')).toMatchObject({ tool: 'Skill', input: 'Copywriting' });
            expect(list.at(-1).reply.steps.map(({ tool, input, failed }) => ({ tool, input, failed: failed || null }))).toEqual([{ tool: 'Skill', input: 'Copywriting', failed: null }]);
            expect(list.at(-1).reply.content).toBe('Here is the headline.');

            // From then on that skill is sent in full and is no longer on the list.
            state.chatCalls.length = 0;
            await send(chat.id, { content: 'another one' });
            expect(state.chatCalls[0].messages[0].content).toContain(copy.text);
            expect(opener(state.chatCalls[0]).function.parameters.properties.skill.enum).toEqual(['Grilling']);

            // A skill called by its own name is opened all the same; a name that is no skill is answered plainly.
            state.chatCalls.length = 0;
            state.toolPlan = [[['Grilling', {}], ['open_skill', { skill: 'nothing here' }]]];
            const { events: more } = await send(chat.id, { content: 'grill me' });
            const answers = state.chatCalls[1].messages.filter((message) => message.role === 'tool');
            expect(answers[0].content.startsWith(grill.text)).toBe(true);
            expect(answers[1].content).toMatch(/^There is no skill by that name in this chat\. The skills are: Grilling\.$/);
            expect(more.at(-1).reply.steps.map(({ tool, input, failed }) => ({ tool, input, failed: failed || null }))).toEqual([{ tool: 'Skill', input: 'Grilling', failed: null }, { tool: 'Skill', input: 'nothing here', failed: 'failed' }]);

            // With every skill opened there is nothing left to list, so the function is gone.
            state.chatCalls.length = 0;
            await send(chat.id, { content: 'thanks' });
            expect((state.chatCalls[0].tools || []).some((tool) => tool.function.name === 'open_skill')).toBe(false);
            expect(state.chatCalls[0].messages[0].content).toContain(grill.text);
        });

        test('skills are linked by what they say about each other, with no model: only names written as skills count', async () => {
            const skill = (id, name, text, source = 'demo') => ({ id: `${source}-${id}`, name, source, text });
            const all = [
                skill('cold-email', 'Cold Email', 'Write the email. For the subject line use the `copywriting` skill.'),
                skill('copywriting', 'Copywriting', 'Pair with the Customer Research skill. See scripts/pr or https://x.example/pr; a PR and a pdf are not skills.'),
                skill('customer-research', 'Customer Research', 'Run /cold-email when the list is ready.'),
                skill('pr', 'PR', 'Opens a pull request.'),
                skill('review', 'Review', 'A review of the week.'),
                skill('weekly', 'Weekly', 'Every week, review what happened.'),
                skill('copywriting', 'Copywriting', 'Another project\'s skill of the same name.', 'other'),
                skill('anatomy', 'Anatomy', 'Draws machines.'),
                skill('hairline-create', 'Hairline Create', 'Line figures.')
            ];
            const links = skillLinks(all, 'Anatomy and Hairline Create together\nAnatomy draws it; Hairline Create moves one part.\n\nA plain question\nNo skill.');
            const names = (id) => links.get(id).map((one) => all.find((s) => s.id === one).name);
            // A backtick name, "the X skill" and a /command all count, and every link goes both ways.
            expect(names('demo-cold-email')).toEqual(expect.arrayContaining(['Copywriting', 'Customer Research']));
            expect(names('demo-copywriting')).toEqual(expect.arrayContaining(['Cold Email', 'Customer Research']));
            // A path, an address, or a short or one-word name used as an ordinary word does not.
            expect(names('demo-pr')).toEqual([]);
            expect(names('demo-review')).toEqual([]);
            // Two skills of the same name are never linked to each other.
            expect(names('other-copywriting')).not.toContain('Copywriting');
            // The combining guide links the skills it names together.
            expect(names('demo-anatomy')).toEqual(['Hairline Create']);

            // On the real library: Mollama's own figure skills are linked through the guide, and no skill has more than six links.
            const real = skillLinks(SKILLS.filter((one) => one.source !== 'mollama'), fs.readFileSync(new URL('../../server/instructions/skills/skill-map/examples/combining.md', import.meta.url), 'utf8'));
            expect(real.get('anatomy-anatomy')).toEqual(expect.arrayContaining(['hairline-hairline-create']));
            expect([...real.values()].every((list) => list.length <= 6)).toBe(true);
        });

        test('find_skill also brings the skills its best matches go with', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.toolPlan = [[['find_skill', { query: 'anatomy' }]]];
            await send(chat.id, { content: 'draw an engine' });
            const answer = state.chatCalls[1].messages.find((message) => message.role === 'tool').content;
            expect(answer).toContain('- Anatomy (');
            expect(answer).toMatch(/- Hairline Create \([^)]+\): .* Goes with Anatomy\./);
        });

        test('a model can look in the skill library for a skill that is not on, and is told it cannot turn one on', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.toolsSeen = [];
            state.toolPlan = [[['find_skill', { query: 'cold email' }], ['find_skill', { query: 'zzzz' }]]];
            state.chatReply = 'Turn on Cold Email under Skills.';
            const { events: list } = await send(chat.id, { content: 'write a cold email' });

            const finder = state.toolsSeen[0].find((tool) => tool.function.name === 'find_skill');
            expect(finder.function.description).toContain(`Search the library of ${SKILLS.length - 4} skills`);
            expect(finder.function.description).toMatch(/You cannot turn a skill on yourself: when one would help, tell the user its name/);
            const answers = state.chatCalls[1].messages.filter((message) => message.role === 'tool');
            const cold = SKILLS.find((skill) => skill.id === 'marketing-cold-email');
            expect(answers[0].content.split('\n')[0]).toMatch(/^These skills are not turned on in this chat\. Tell the user by name which one would help/);
            expect(answers[0].content.split('\n')[1]).toBe(`- Cold Email (Marketing (Corey Haines)): ${cold.description}`);
            // Five matches at most, and up to three skills the best two go with.
            expect(answers[0].content.split('\n').length).toBeLessThanOrEqual(9);
            expect(answers[0].content).not.toContain(cold.text.slice(40, 200));
            expect(answers[1].content).toBe('No skill in the library matches. Answer without one.');
            expect(list.at(-1).reply.steps.map(({ tool, input }) => ({ tool, input }))).toEqual([{ tool: 'Skill library', input: 'cold email' }, { tool: 'Skill library', input: 'zzzz' }]);
            // Looking is all it does: the chat's skills are unchanged.
            expect((await get(`/chats/${chat.id}`)).data.chat.skills).toEqual(['components', 'skill-map', 'richmd-rich-md-first', 'explainer-visual-explainer']);

            // A skill that is already on is not offered again by the search.
            state.toolsSeen = [];
            state.chatCalls.length = 0;
            state.toolPlan = [[['find_skill', { query: 'cold email' }]]];
            const has = await newChat({ model: 'qwen2.5-coder:7b', skills: ['components', 'marketing-cold-email'] });
            await send(has.id, { content: 'write a cold email' });
            expect(state.chatCalls[1].messages.find((message) => message.role === 'tool').content).not.toContain('- Cold Email (');

            // Whole words are matched, the name counts most, and filler words are ignored.
            const names = (query) => rankSkills(SKILLS, query).map((skill) => skill.name);
            expect(names('cold email for bakery owner delivery app')[0]).toBe('Cold Email');
            expect(names('app launch planning')[0]).toMatch(/Launch/);
            expect(names('app launch planning')).not.toContain('Webapp Testing');
            expect(names('app launch planning')).not.toContain('Apple Mail Setup');
            expect(names('grill me on my plans')[0]).toBe('Grilling');
            expect(names('help with the new')).toEqual([]);
            expect(names('')).toEqual([]);

            // A model that cannot use tools is offered nothing.
            state.toolsSeen = [];
            state.noTools = true;
            const plain = await newChat({ model: 'qwen2.5-coder:7b' });
            await send(plain.id, { content: 'hello' });
            state.noTools = false;
            expect(state.toolsSeen).toEqual([null]);
        });

        test('a skill may ship files: they are listed, read by their path only, and handed to a model one at a time', async () => {
            const hairline = SKILLS.find((skill) => skill.id === 'hairline-hairline-create');
            const onDisk = (name) => fs.readFileSync(new URL(`../../server/instructions/library/hairline/hairline-hairline-create/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
            const paths = hairline.files.map((entry) => entry.path);
            expect(paths).toEqual(['bench.html', 'build.mjs', 'concepts.md', 'examples/riffle.js', 'examples/terrain.js', 'kernel.js', 'look.md', 'look.mjs', 'read.mjs', 'rules.md', 'source.md', 'validate.mjs']);
            // The kit is its project's own, unchanged: the engine carries the mark its authors put on it.
            expect(onDisk('kernel.js')).toMatch(/^\/\* hairline kernel sha256:[0-9a-f]{64} \*\//);
            // Each skill that ships files has them beside its page, and every file it lists is there to read.
            const withFiles = SKILLS.filter((skill) => (skill.files || []).length > 0);
            expect(withFiles.map((skill) => skill.id)).toEqual(expect.arrayContaining(['skill-map', 'anatomy-anatomy', 'hairline-hairline-create', 'richmd-rich-md-first', 'explainer-visual-explainer', 'marketing-cold-email', 'engineering-tdd', 'anthropic-internal-comms', 'dex-dhh-rails-style']));
            expect(withFiles.length).toBeGreaterThan(140);
            for (const skill of withFiles) for (const file of skill.files) expect(fs.existsSync(`${skill.folder}/${file.path}`)).toBe(true);

            // The page that shows a figure asks for the files by path.
            expect((await get('/skills')).data.skills.find((skill) => skill.id === hairline.id).files).toEqual(paths);
            const fileOf = (id, path) => client.get(`/api/v1/skills/${id}/file?path=${encodeURIComponent(path)}`, { validateStatus: () => true });
            const bench = await fileOf(hairline.id, 'bench.html');
            expect(bench.status).toBe(200);
            expect(bench.data).toMatchObject({ success: true, skill: hairline.id, path: 'bench.html', text: onDisk('bench.html') });
            expect(bench.data.text).toContain('<script id="hl-kernel">/*KERNEL*/</script>');
            expect(bench.data.text).toContain('/*FIGURE*/');
            expect((await fileOf(hairline.id, 'examples/terrain.js')).data.text).toBe(onDisk('examples/terrain.js'));
            // Nothing but a listed file is ever read: no way up and out of the folder, and no other skill's page.
            for (const outside of ['../hairline-hairline-create.md', '../../THIRD_PARTY.md', 'examples/../../hairline-hairline-adopt.md', 'nothing.js', '/etc/passwd', 'kernel.js/..']) {
                expect((await fileOf(hairline.id, outside)).status, outside).toBe(404);
            }
            expect((await fileOf(hairline.id, 'a<b>.js')).status).toBe(400);
            expect((await fileOf('engineering-grilling', 'kernel.js')).status).toBe(404);
            expect((await fileOf('not-a-skill', 'kernel.js')).status).toBe(404);

            // A model reads one file at a time, when the skill's own page sends it there.
            const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: ['components', hairline.id] });
            state.chatCalls.length = 0;
            state.toolPlan = [[['open_skill_file', { skill: hairline.id, file: 'rules.md' }], ['open_skill_file', { skill: 'Hairline Create', file: 'missing.md' }]]];
            const { events: list } = await send(chat.id, { content: 'draw a figure' });
            const reader = state.chatCalls[0].tools.find((tool) => tool.function.name === 'open_skill_file');
            expect(reader.function.description).toContain(`\n- ${hairline.id}: ${paths.join(', ')}`);
            expect(reader.function.parameters.properties.skill.enum).toEqual([hairline.id]);
            // Room is kept for a file to be read without pushing the conversation out.
            expect(state.chatCalls[0].options.num_ctx).toBeGreaterThanOrEqual(16384);
            const answers = state.chatCalls[1].messages.filter((message) => message.role === 'tool');
            expect(answers[0].content).toBe(onDisk('rules.md'));
            expect(answers[1].content).toMatch(/^That skill has no file by that name\. Its files are: bench\.html, build\.mjs, /);
            expect(list.at(-1).reply.steps.map(({ tool, input, failed }) => ({ tool, input, failed: failed || null }))).toEqual([
                { tool: 'Skill file', input: 'rules.md (Hairline Create)', failed: null },
                { tool: 'Skill file', input: 'missing.md (Hairline Create)', failed: 'failed' }
            ]);

            // A chat whose skills have no files is not offered the function, and neither is a model that cannot use tools.
            state.chatCalls.length = 0;
            const bare = SKILLS.find((skill) => skill.source !== 'mollama' && !(skill.files || []).length);
            const plain = await newChat({ model: 'qwen2.5-coder:7b', skills: ['components', bare.id] });
            await send(plain.id, { content: 'hello' });
            expect(state.chatCalls[0].tools.some((tool) => tool.function.name === 'open_skill_file')).toBe(false);

            client.clearCookies();
            client.authenticatedAs = null;
            expect((await fileOf(hairline.id, 'bench.html')).status).toBe(401);
            await testStartup.loginAsUser('user');
        });

        test('the Hairline build makes the checks of the skill\'s validate command: a figure with something to fix goes back once before it is shown', async () => {
            const onDisk = (name) => fs.readFileSync(new URL(`../../server/instructions/library/hairline/hairline-hairline-create/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
            const riffle = onDisk('examples/riffle.js').trim();
            // A clock of its own and a colour of its own, both against the skill's rules.
            const wrong = riffle.replace(/\bhairline\s*\(\s*\{(?![\s\S]*\bhairline\s*\(\s*\{)/, (found) => `setTimeout(() => {}, 10);\nconst tint = "#ff0000";\n${found}`);
            expect(wrong).not.toBe(riffle);
            const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: ['components', 'hairline-hairline-create'] });

            // Both of the kit's own examples pass, so the checks are not stricter than the skill's.
            state.chatCalls.length = 0;
            state.toolPlan = [[['hairline_build', { code: riffle }]]];
            await send(chat.id, { content: 'draw the riffle' });
            expect(state.chatCalls[1].messages.at(-1).content).toMatch(/^Built hairline-riffle\.html and showed it to the user in the chat\. validate: ok/);

            // The first build of a faulty figure shows nothing and says what to fix, in the validator's own words.
            state.chatCalls.length = 0;
            state.toolPlan = [[['hairline_build', { code: wrong }]], [['hairline_build', { code: riffle }]]];
            const mended = await send(chat.id, { content: 'draw it again' });
            const told = state.chatCalls[1].messages.at(-1).content;
            expect(told).toMatch(/^Not shown yet\. validate found 2 to fix in hairline-riffle\.html:\npaint: rule 04\. /);
            expect(told).toContain('\nclock: rule 07. The figure runs a clock of its own.');
            expect(told.endsWith('Fix these in the figure and call hairline_build again with the whole file.')).toBe(true);
            expect(state.chatCalls[2].messages.at(-1).content).toMatch(/validate: ok/);
            expect(mended.events.at(-1).reply.content.match(/<Frame /g)).toHaveLength(1);

            // Built once with faults and never again: the figure held back is shown when the reply ends.
            state.chatCalls.length = 0;
            state.toolPlan = [[['hairline_build', { code: wrong }]]];
            state.chatReply = 'I could not finish the figure.';
            const left = await send(chat.id, { content: 'one more' });
            expect(left.events.at(-1).reply.content.match(/<Frame /g)).toHaveLength(1);
            expect(left.events.at(-1).reply.content).toContain('I could not finish the figure.');

            // Sent again unfixed, it is shown all the same with what is still wrong, so the person is not left with nothing.
            state.chatCalls.length = 0;
            state.toolPlan = [[['hairline_build', { code: wrong }]], [['hairline_build', { code: wrong }]]];
            const stuck = await send(chat.id, { content: 'and again' });
            expect(state.chatCalls[2].messages.at(-1).content).toMatch(/^Built hairline-riffle\.html and showed it to the user in the chat, with 2 still to fix:\npaint: /);
            expect(stuck.events.at(-1).reply.content.match(/<Frame /g)).toHaveLength(1);
            state.toolPlan = [];
        });

        test('what a skill does with a command, Mollama does itself: the Hairline build shows the figure in the chat', async () => {
            const onDisk = (name) => fs.readFileSync(new URL(`../../server/instructions/library/hairline/hairline-hairline-create/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
            const terrain = onDisk('examples/terrain.js').trim();
            const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: ['components', 'hairline-hairline-create'] });
            state.chatCalls.length = 0;
            state.toolPlan = [
                [['open_skill_file', { skill: 'hairline-hairline-create', file: 'kernel.js', until: 'var HL' }]],
                [['hairline_build', { code: terrain }]],
                [['hairline_build', { code: 'const x = 1;' }]]
            ];
            state.chatReply = 'Here is the terrain.';
            const { events: list } = await send(chat.id, { content: 'draw the terrain' });

            // The function stands in for the skill's own build command, and says so.
            const builder = state.chatCalls[0].tools.find((tool) => tool.function.name === 'hairline_build');
            expect(builder.function.description).toMatch(/^Does what `node build\.mjs <name>\.js` and `node validate\.mjs` do in the Hairline skill, without Node/);
            expect(Object.keys(builder.function.parameters.properties)).toEqual(['code']);

            // A file can be read only as far as the instructions ask: here the engine's index, not the code under it.
            const head = state.chatCalls[1].messages.at(-1).content;
            expect(head).toMatch(/^\/\* hairline kernel sha256:/);
            expect(head).toContain('Cam(azDeg, k, S)');
            expect(head).not.toContain('var HL');
            expect(head.length).toBeLessThan(onDisk('kernel.js').length / 2);

            // The figure is put on the skill's own page and shown in the reply; a file that is not a figure builds nothing.
            expect(state.chatCalls[2].messages.at(-1)).toMatchObject({ role: 'tool', content: 'Built hairline-terrain.html and showed it to the user in the chat. validate: ok, the figure passes the static checks. You cannot see the page from here, so the look has not been done.' });
            expect(state.chatCalls[3].messages.at(-1).content).toMatch(/^Nothing was built\. A figure file ends with hairline\(/);
            const reply = list.at(-1).reply;
            expect(reply.content).toContain(['```component', '<Frame template="hairline-hairline-create/bench.html" height="640px" title="Hairline terrain" palette="ground:background, ink:text, muted:text 65%, line:border, hairline-plate:background, hairline-hi:text, hairline-edge:text 55%, hairline-mid:text 32%, hairline-lo:border">', '<Part name="KERNEL" file="hairline-hairline-create/kernel.js" />', '<Part name="FIGURE">', terrain, '</Part>', '</Frame>', '```'].join('\n'));
            expect(reply.content.match(/<Frame /g)).toHaveLength(1);
            // While each tool runs the person is told what is being done, and told nothing stale once it is over.
            const notes = list.filter((event) => event.type === 'note').map((event) => event.text);
            expect(notes).toEqual(["Reading one of a skill's files…", '', 'Building the figure…', '', 'Building the figure…', '']);
            expect(reply.content.trim().endsWith('Here is the terrain.')).toBe(true);
            expect(reply.steps.map(({ tool, input, failed }) => ({ tool, input, failed: failed || null }))).toEqual([
                { tool: 'Skill file', input: 'kernel.js (Hairline Create)', failed: null },
                { tool: 'Hairline build', input: 'Hairline Create', failed: null },
                { tool: 'Hairline build', input: 'Hairline Create', failed: 'failed' }
            ]);

            // What the reply read through its tools is not counted as part of the chat afterwards.
            const after = (await get(`/chats/${chat.id}`)).data.chat.context;
            expect(after.usedTokens).toBeLessThan(estimateTokens(onDisk('kernel.js')));

            // A chat without the skill has no such function.
            state.chatCalls.length = 0;
            const plain = await newChat({ model: 'qwen2.5-coder:7b' });
            await send(plain.id, { content: 'hello' });
            expect(state.chatCalls[0].tools).toBeUndefined();
        });

        test('with sandboxes set up, the Hairline checks are the kit\'s own scripts run in one; when it cannot start or is slow, the checks kept here are made', async () => {
            const riffle = fs.readFileSync(new URL('../../server/instructions/library/hairline/hairline-hairline-create/examples/riffle.js', import.meta.url), 'utf8').replace(/\r\n/g, '\n').trim();
            const wrong = riffle.replace(/\bhairline\s*\(\s*\{(?![\s\S]*\bhairline\s*\(\s*\{)/, (found) => `setTimeout(() => {}, 10);\n${found}`);
            const uploads = [];
            const ended = [];
            let mode = 'faulty';
            let finishSlow = () => {};
            process.env.SANDBOX_API_KEY = 'rnd_test';
            process.env.SANDBOX_WORKSPACE_ID = 'tea-test';
            process.env.SANDBOX_SNAPSHOT = 'mollama-skills';
            setSandboxClient({
                create: async () => {
                    if (mode === 'down') throw new Error('quota reached');
                    return { id: 'sbx-hairline', status: 'running' };
                },
                exec: async (id, line) => {
                    let events = [{ type: 'exit', exit_code: 0 }];
                    if (line.includes('timeout')) {
                        if (mode === 'slow') await new Promise((resolve) => { finishSlow = resolve; });
                        events = mode === 'clean'
                            ? [{ type: 'output', stream: 'stdout', data: 'ok page.html: kernel and bench intact, and the figure passes the static checks.\n' }, { type: 'exit', exit_code: 0 }]
                            : [{ type: 'output', stream: 'stderr', data: 'spin: said by the kit in the sandbox only.\n1 to fix in page.html\n' }, { type: 'exit', exit_code: 1 }];
                    }
                    return (async function* () { yield* events; })();
                },
                upload: async (id, where, data) => { uploads.push({ where, data: Buffer.from(data).toString() }); },
                download: async () => ({ data: Buffer.from('') }),
                terminate: async (id) => { ended.push(id); }
            });
            setHairlineWait(300);
            try {
                const build = async (code) => {
                    const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: ['components', 'hairline-hairline-create'] });
                    state.chatCalls.length = 0;
                    state.toolPlan = [[['hairline_build', { code }]]];
                    await send(chat.id, { content: 'draw it' });
                    return state.chatCalls[1].messages.at(-1).content;
                };

                // The figure goes in as figure.js; build.mjs puts it on the bench and validate.mjs checks the page, both
                // from the skill's own folder; what validate prints is what the model is told, and the sandbox goes.
                const told = await build(riffle);
                expect(uploads.find((one) => one.where === '/root/work/figure.js').data).toBe(riffle + '\n');
                const job = uploads.find((one) => one.where === '/root/job/main.sh').data;
                expect(job).toContain('node /root/skills/hairline-create/build.mjs figure.js page.html');
                expect(job).toContain('node /root/skills/hairline-create/validate.mjs page.html');
                expect(ended).toEqual(['sbx-hairline']);
                expect(told).toBe('Not shown yet. validate found 1 to fix in hairline-riffle.html:\nspin: said by the kit in the sandbox only.\nFix these in the figure and call hairline_build again with the whole file.');

                // A page validate passes is shown.
                mode = 'clean';
                expect(await build(riffle)).toMatch(/^Built hairline-riffle\.html and showed it to the user in the chat\. validate: ok/);

                // A sandbox that cannot start, or is slow, is not waited on: the checks kept here speak instead.
                for (const which of ['down', 'slow']) {
                    mode = which;
                    const local = await build(wrong);
                    expect(local).toMatch(/^Not shown yet\. validate found 1 to fix/);
                    expect(local).toMatch(/clock: rule 07/);
                    expect(local).not.toMatch(/said by the kit in the sandbox only/);
                }
            } finally {
                finishSlow();
                setHairlineWait(null);
                setSandboxClient(null);
                delete process.env.SANDBOX_API_KEY;
                delete process.env.SANDBOX_WORKSPACE_ID;
                delete process.env.SANDBOX_SNAPSHOT;
            }
        });

        test('a program the model writes runs in a sandbox with no network, on the person\'s files, once they allow it, and the sandbox is always thrown away', async () => {
            const made = [];
            const uploads = [];
            const lines = [];
            const sent = [];
            const ended = [];
            let fault = null;
            let madeList = 'out/summary.pdf\nnotes.txt\n';
            process.env.SANDBOX_API_KEY = 'rnd_test';
            process.env.SANDBOX_WORKSPACE_ID = 'tea-test';
            process.env.SANDBOX_SNAPSHOT = 'mollama-skills';
            setSandboxClient({
                create: async (options) => {
                    if (fault === 'start') throw new Error('quota reached');
                    made.push(options);
                    return { id: `sbx-${made.length}`, status: 'running' };
                },
                exec: async (id, line) => {
                    lines.push(line);
                    if (fault === 'run' && line.includes('timeout')) throw new Error('stream lost');
                    if (fault === 'gone' && id === 'sbx-1') throw new Error('sandbox not found');
                    const events = line.includes('timeout') ? [{ type: 'output', stream: 'stdout', data: 'made it\n' }, { type: 'exit', exit_code: 0 }]
                        : line.startsWith('cd /root/work && find') ? [{ type: 'output', stream: 'stdout', data: madeList }, { type: 'exit', exit_code: 0 }]
                            : [{ type: 'exit', exit_code: 0 }];
                    return (async function* () { yield* events; })();
                },
                upload: async (id, where, data) => { sent.push({ id, where, data: Buffer.from(data).toString() }); },
                download: async (id, where) => ({ data: Buffer.from(`content of ${where}`) }),
                terminate: async (id) => { ended.push(id); }
            });
            setFilesFetch(async (url, options) => {
                if (!String(url).endsWith('/files/upload')) return fakeFilesystemOne(url, options);
                uploads.push({ folder: options.body.get('basePath'), paths: options.body.get('relativePaths'), conflict: options.body.get('onConflict') });
                return json(201, { success: true });
            });
            try {
                const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: ['anthropic-pdf'] });
                const turn = async (input, answer) => {
                    state.toolPlan = [[['run_program', input]]];
                    state.chatCalls = [];
                    const pending = send(chat.id, { content: 'make a summary' });
                    for (let tries = 0; tries < 200 && !waitingPermissions().some((waiting) => waiting.chatId === chat.id); tries += 1) await wait(25);
                    const waiting = waitingPermissions().find((one) => one.chatId === chat.id);
                    expect(waiting).toBeTruthy();
                    expect((await post(`/chats/${chat.id}/permission`, { id: waiting.id, answer })).status).toBe(200);
                    await pending;
                    return state.chatCalls[1].messages.at(-1).content;
                };
                const work = { language: 'python', code: 'print("made it")', files: ['/docs/report.pdf'] };

                // It changes the person's files, so it waits for them; refused, no sandbox is started.
                await turn(work, 'no');
                expect(made).toHaveLength(0);

                // Allowed: a sandbox starts from the prepared snapshot with the network shut, the person's file and the
                // program go in, the program runs, what it made comes back and is saved beside the file, and the sandbox goes.
                const told = await turn(work, 'once');
                expect(made).toEqual([expect.objectContaining({ snapshotName: 'mollama-skills', networkPolicy: { type: 'deny-all' } })]);
                expect(sent.map((one) => one.where)).toEqual(['/root/work/report.pdf', '/root/job/main.py']);
                expect(sent[0].data).toBe(fsFiles['/docs/report.pdf'].bytes.toString());
                expect(sent[1].data).toBe('print("made it")');
                expect(lines.find((line) => line.includes('timeout'))).toMatch(/^cd \/root\/work && export .*timeout -k 5 300 python \/root\/job\/main\.py$/);
                expect(lines.join('\n')).not.toMatch(/rnd_test|tea-test/);
                expect(uploads).toEqual([{ folder: '/docs', paths: '["out/summary.pdf","notes.txt"]', conflict: 'rename' }]);
                // The sandbox is kept for this chat, so what the run left is there for the next one.
                expect(ended).toEqual([]);
                expect(told).toMatch(/^The python program finished with exit code 0\.\n\nIt printed:\nmade it/);
                expect(told).toMatch(/Saved in \/docs: out\/summary\.pdf, notes\.txt\./);
                const offered = state.chatCalls[0].tools.find((tool) => tool.function.name === 'run_program');
                expect(offered.function.description).toMatch(/no network/);
                expect(offered.function.description).toMatch(/PDF in \/root\/skills\/pdf/);
                expect(offered.function.parameters.properties.minutes).toMatchObject({ minimum: 1, maximum: 30 });

                // The next run in the chat uses the same sandbox, and may be given longer to run.
                lines.length = 0;
                expect(await turn({ ...work, minutes: 20 }, 'once')).toMatch(/^The python program finished/);
                expect(made).toHaveLength(1);
                expect(lines.find((line) => line.includes('timeout'))).toMatch(/timeout -k 5 1200 python/);

                // A kept sandbox that has gone is replaced, and the model is told the earlier files are gone.
                fault = 'gone';
                expect(await turn(work, 'once')).toMatch(/^This run started on a fresh computer: the files earlier runs left are gone\.\n\nThe python program finished/);
                expect(made).toHaveLength(2);
                expect(ended).toEqual(['sbx-1']);
                fault = null;

                // Left alone, a kept sandbox is let go.
                await releaseSandboxes();
                expect(ended).toEqual(['sbx-1', 'sbx-2']);

                // A sandbox that cannot start, and a run that breaks, each say nothing was run.
                fault = 'start';
                expect(await turn(work, 'once')).toBe('Nothing was run. A sandbox could not be started: quota reached');
                fault = 'run';
                expect(await turn(work, 'once')).toBe('Nothing was run. stream lost');
                await releaseSandboxes();
                expect(ended).toEqual(['sbx-1', 'sbx-2', 'sbx-3']);
                fault = null;
                expect(await turn({ language: 'ruby', code: 'puts 1', files: ['/docs/report.pdf'] }, 'once')).toMatch(/^Nothing was run\. Say what the program is written in/);

                // A program given no files and no folder reads nothing of the person's and saves nothing, so it runs
                // without asking; what it makes is reported, not saved.
                const before = uploads.length;
                state.toolPlan = [[['run_program', { language: 'bash', code: 'ls /root/skills' }]]];
                state.chatCalls = [];
                const { events: looked } = await send(chat.id, { content: 'look around' });
                expect(looked.some((event) => event.type === 'permission')).toBe(false);
                expect(state.chatCalls[1].messages.at(-1).content).toMatch(/^The bash program finished with exit code 0\.[\s\S]*nothing was saved: call again with save_to/);
                expect(uploads).toHaveLength(before);

                // Pictures named in look come back to a model that can see, as images on the result, and are not saved;
                // one that was not made is said to be missing.
                madeList = 'shots/front.png\nfigure.html\n';
                state.vision = true;
                state.toolPlan = [[['run_program', { language: 'bash', code: 'chromium --headless --screenshot=shots/front.png figure.html', look: ['shots/front.png', 'shots/side.png'] }]]];
                state.chatCalls = [];
                await send(chat.id, { content: 'build it and look' });
                const seenResult = state.chatCalls[1].messages.at(-1);
                expect(seenResult.images).toEqual([Buffer.from('content of /root/work/shots/front.png').toString('base64')]);
                expect(seenResult.content).toMatch(/The pictures you asked to see are attached: shots\/front\.png\. Not attached \(not made[^)]*\): shots\/side\.png\./);
                expect(uploads).toHaveLength(before);

                // A model that cannot see is told the pictures were not sent.
                state.vision = false;
                state.toolPlan = [[['run_program', { language: 'bash', code: 'chromium --headless --screenshot=shots/front.png figure.html', look: ['shots/front.png'] }]]];
                state.chatCalls = [];
                await send(chat.id, { content: 'look again' });
                expect(state.chatCalls[1].messages.at(-1).images).toBeUndefined();
                expect(state.chatCalls[1].messages.at(-1).content).toMatch(/This model cannot see pictures, so they were not sent\.$/);
                madeList = 'out/summary.pdf\nnotes.txt\n';

                // With sandboxes set up, a skill whose files are in the snapshot lacks nothing.
                const lacks = async (id) => (await get('/skills')).data.skills.find((skill) => skill.id === id).needs;
                expect(await lacks('anthropic-pdf')).toEqual([]);
                expect(await lacks('anatomy-anatomy')).toEqual([]);

                // Without the settings, nothing is offered, and the skills say what they lack again.
                delete process.env.SANDBOX_SNAPSHOT;
                state.toolPlan = [];
                state.chatCalls = [];
                await send(chat.id, { content: 'hello' });
                expect((state.chatCalls[0].tools || []).some((tool) => tool.function.name === 'run_program')).toBe(false);
                expect(await lacks('anatomy-anatomy')).toEqual(['commands']);
            } finally {
                await releaseSandboxes();
                setSandboxClient(null);
                setFilesFetch(fakeFilesystemOne);
                delete process.env.SANDBOX_API_KEY;
                delete process.env.SANDBOX_WORKSPACE_ID;
                delete process.env.SANDBOX_SNAPSHOT;
            }
        });

        test('Visual Explainer renders in a sandbox, goes back once to be fixed, shows in the app\'s theme with its kit named not pasted, and a long page is not sent back to the model', async () => {
            const kitFile = (name) => fs.readFileSync(new URL(`../../server/instructions/library/explainer/explainer-visual-explainer/kit/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
            const css = kitFile('kit.css');
            const builtPage = (body) => `<!doctype html><html><head><style data-ve-kit>\n${css}</style></head><body><main>${body}</main><script data-ve-kit>\n${[kitFile('reply.js'), kitFile('kit.js')].join('\n')}</script></body></html>`;
            let body = '<h1>94% of reads skip Postgres</h1>';
            let renders = 0;
            const lines = [];
            const uploaded = [];
            const started = [];
            process.env.SANDBOX_API_KEY = 'rnd_test';
            process.env.SANDBOX_WORKSPACE_ID = 'tea-test';
            process.env.SANDBOX_SNAPSHOT = 'mollama-skills';
            setSandboxClient({
                create: async (options) => { started.push(options); return { id: 'sbx-ve', status: 'running' }; },
                exec: async (id, line) => {
                    lines.push(line);
                    if (line.includes('timeout')) {
                        renders += 1;
                        const events = renders === 1
                            ? [{ type: 'output', stream: 'stdout', data: '✓ page.html  kit inlined\n  ✗ a label is cut off at 390px\n✗ 1 visual problem(s); the page was written. Fix them and render again.\n' }, { type: 'exit', exit_code: 1 }]
                            : [{ type: 'output', stream: 'stdout', data: '✓ page.html  kit inlined\n  ✓ Visual check: no problems\n' }, { type: 'exit', exit_code: 0 }];
                        return (async function* () { yield* events; })();
                    }
                    const events = line.startsWith('cd /root/work && find') ? [{ type: 'output', stream: 'stdout', data: 'page.html\n' }, { type: 'exit', exit_code: 0 }] : [{ type: 'exit', exit_code: 0 }];
                    return (async function* () { yield* events; })();
                },
                upload: async (id, where, data) => { uploaded.push({ where, text: Buffer.from(data).toString() }); },
                download: async () => ({ data: Buffer.from(builtPage(body)) }),
                terminate: async () => {}
            });
            try {
                // On by default for a model that can use tools, and offered only where its render can run.
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const source = '<!doctype html><html><head><link rel="stylesheet" href="kit.css" data-ve-kit></head><body><main><h1>Cache</h1></main></body></html>';
                state.chatCalls.length = 0;
                state.toolPlan = [[['visual_explainer_render', { page: source, title: 'Cache reads' }]], [['visual_explainer_render', { page: source, title: 'Cache reads' }]]];
                state.chatReply = 'Here is the page.';
                await send(chat.id, { content: 'explain the cache' });
                expect(uploaded.find((one) => one.where === '/root/work/page.src.html').text).toBe(source);
                // A render may reach the two addresses its page's fonts come from, and nothing else.
                expect(started[0].networkPolicy).toEqual({ type: 'allow-list', rules: [{ domain: 'fonts.googleapis.com' }, { domain: 'fonts.gstatic.com' }] });
                expect(uploaded.find((one) => one.where === '/root/job/main.sh').text).toMatch(/^node \/root\/skills\/visual-explainer\/kit\/render\.mjs page\.src\.html -o page\.html/);
                expect(state.chatCalls[1].messages.at(-1).content).toMatch(/^Not shown yet\. The render found something to fix:\n[\s\S]*a label is cut off at 390px[\s\S]*Fix these and call visual_explainer_render again/);
                expect(state.chatCalls[2].messages.at(-1).content).toMatch(/^Rendered and showed "Cache reads" to the user in the chat, in the app's theme\. Report:/);

                // The reply holds the page with the kit's files named, and the theme in place of the kit's colours.
                const shown = (await get(`/chats/${chat.id}`)).data.chat.messages.at(-1).content;
                expect(shown).toMatch(/```component\n<Frame height="720px" title="Cache reads" palette="bg:background, surface:surface, [^"]*accent:primary[^"]*">/);
                expect(shown).toContain('<Part name="VE_CSS_1" file="explainer-visual-explainer/kit/kit.css" />');
                expect(shown).toContain('<Part name="VE_JS_1" file="explainer-visual-explainer/kit/reply.js" />');
                expect(shown).toContain('<Part name="VE_JS_2" file="explainer-visual-explainer/kit/kit.js" />');
                expect(shown).toContain('<style data-ve-kit>/*VE_CSS_1*/</style>');
                expect(shown).not.toContain(css.slice(0, 300));
                expect(shown.length).toBeLessThan(2000);

                // A long page is shown whole to the person but sent back to the model as one line.
                body = 'x'.repeat(8000);
                renders = 1;
                state.toolPlan = [[['visual_explainer_render', { page: source, title: 'Long page' }]]];
                await send(chat.id, { content: 'a longer one' });
                state.toolPlan = [];
                state.chatCalls.length = 0;
                await send(chat.id, { content: 'thanks' });
                const history = state.chatCalls[0].messages.filter((message) => message.role === 'assistant').map((message) => message.content).join('\n');
                expect(history).toMatch(/\[Shown here: "Long page", a page of [\d,]+ characters, not repeated\.\]/);
                expect(history).not.toContain('x'.repeat(8000));
                expect(history).toContain('title="Cache reads"');

                // With sandboxes set up it lacks nothing; without them it is not offered and says it needs commands.
                const lacks = async () => (await get('/skills')).data.skills.find((skill) => skill.id === 'explainer-visual-explainer').needs;
                expect(await lacks()).toEqual([]);
                delete process.env.SANDBOX_SNAPSHOT;
                state.chatCalls.length = 0;
                await send(chat.id, { content: 'hello' });
                expect(state.toolsSeen.at(-1).some((tool) => tool.function.name === 'visual_explainer_render')).toBe(false);
                expect(await lacks()).toEqual(['commands']);
                expect((await get('/skills')).data.skills.find((skill) => skill.id === 'richmd-rich-md-first').needs).toEqual([]);
            } finally {
                setSandboxClient(null);
                delete process.env.SANDBOX_API_KEY;
                delete process.env.SANDBOX_WORKSPACE_ID;
                delete process.env.SANDBOX_SNAPSHOT;
            }
        });

        test('an office skill\'s own scripts run as a program in the sandbox, beside the skill\'s files, and what they make is saved in the person\'s own folder', async () => {
            const uploads = [];
            const lines = [];
            process.env.SANDBOX_API_KEY = 'rnd_test';
            process.env.SANDBOX_WORKSPACE_ID = 'tea-test';
            process.env.SANDBOX_SNAPSHOT = 'mollama-skills';
            setSandboxClient({
                create: async () => ({ id: 'sbx-office', status: 'running' }),
                exec: async (id, line) => {
                    lines.push(line);
                    const events = line.includes('timeout') ? [{ type: 'output', stream: 'stdout', data: 'Found 2 fields\n' }, { type: 'exit', exit_code: 0 }]
                        : line.startsWith('cd /root/work && find') ? [{ type: 'output', stream: 'stdout', data: 'fields.json\n' }, { type: 'exit', exit_code: 0 }]
                            : [{ type: 'exit', exit_code: 0 }];
                    return (async function* () { yield* events; })();
                },
                upload: async () => {},
                download: async () => ({ data: Buffer.from('{"a":1}') }),
                terminate: async () => {}
            });
            setFilesFetch(async (url, options) => {
                if (!String(url).endsWith('/files/upload')) return fakeFilesystemOne(url, options);
                uploads.push({ folder: options.body.get('basePath'), paths: options.body.get('relativePaths') });
                return json(201, { success: true });
            });
            try {
                const chat = await newChat({ model: 'qwen2.5-coder:7b', skills: ['anthropic-pdf'] });
                const turn = async (input) => {
                    state.toolPlan = [[['run_program', input]]];
                    state.chatCalls = [];
                    const pending = send(chat.id, { content: 'check the form' });
                    for (let tries = 0; tries < 200 && !waitingPermissions().some((waiting) => waiting.chatId === chat.id); tries += 1) await wait(25);
                    await post(`/chats/${chat.id}/permission`, { id: waitingPermissions().find((one) => one.chatId === chat.id).id, answer: 'once' });
                    await pending;
                    return state.chatCalls[1].messages.at(-1).content;
                };
                const work = { language: 'bash', code: 'python /root/skills/pdf/scripts/extract_form_field_info.py report.pdf fields.json', files: ['/docs/report.pdf'] };

                // There is no script function of its own any more: run_program is offered, and says where the skill's files are.
                const told = await turn(work);
                const names = state.chatCalls[0].tools.map((tool) => tool.function.name);
                expect(names).toContain('run_program');
                expect(names.some((name) => name.endsWith('_script'))).toBe(false);
                expect(state.chatCalls[0].tools.find((tool) => tool.function.name === 'run_program').function.description).toContain('PDF in /root/skills/pdf');
                expect(lines.find((line) => line.includes('timeout'))).toMatch(/timeout -k 5 300 bash \/root\/job\/main\.sh$/);
                expect(told).toMatch(/It printed:\nFound 2 fields/);
                // Saved beside the file it was given, or, for a folder the model names, inside the person's own folder.
                expect(uploads.at(-1)).toEqual({ folder: '/docs', paths: '["fields.json"]' });
                expect(await turn({ ...work, save_to: '/Gearbox' })).toMatch(/Saved in \/[a-z0-9]+\/Gearbox: fields\.json\./);
                expect(uploads.at(-1).folder).toMatch(/^\/[a-z0-9]+\/Gearbox$/);

                // With sandboxes set up the skill lacks nothing.
                expect((await get('/skills')).data.skills.find((skill) => skill.id === 'anthropic-pdf').needs).not.toContain('commands');
            } finally {
                await releaseSandboxes();
                setSandboxClient(null);
                setFilesFetch(fakeFilesystemOne);
                delete process.env.SANDBOX_API_KEY;
                delete process.env.SANDBOX_WORKSPACE_ID;
                delete process.env.SANDBOX_SNAPSHOT;
            }
        });

        test('what a model draws is checked by rule, and it is told why and given one chance to put it right', async () => {
            const page = (name) => COMPONENT_PAGES.find((one) => one.name === name).text;
            const block = (inside) => ['Here you go:', '', '```component', inside, '```'].join('\n');
            const bad = block('<Card>\n<Grid>\n<Typography>Engine</Typography>\n</Grid>\n<FloatingActionButton />\n<Video src="engine.mp4" />\n</Card>');
            const good = block('<Card>\n<Typography>Engine</Typography>\n</Card>');

            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.chatCalls.length = 0;
            state.replyPlan = [bad, good];
            const { events: list } = await send(chat.id, { content: 'show the engine' });

            // The same model is asked once more, with the reasons and the pages of the components it used.
            expect(state.chatCalls).toHaveLength(2);
            expect(state.chatCalls[1].model).toBe(state.chatCalls[0].model);
            const [wrote, told] = state.chatCalls[1].messages.slice(-2);
            expect(wrote).toEqual({ role: 'assistant', content: bad });
            expect(told.role).toBe('user');
            expect(told.content).toMatch(/^The chat could not draw part of your reply:\n- "Grid" is not a component\. The components are: Audio, Badge, /);
            expect(told.content).toContain('\n- FloatingActionButton needs the attribute "icon".');
            expect(told.content).toContain('\n- Video: src must be a full https address of a real file, not "engine.mp4". Do not make an address up.');
            // The pages come smallest first, as many as fit a small model: here all but the long Card page.
            for (const name of ['Typography', 'FloatingActionButton', 'Video']) expect(told.content).toContain(page(name));
            expect(told.content).not.toContain(page('Flow'));
            expect(told.content.endsWith('Write your whole reply again with this put right. Do not mention this correction.')).toBe(true);

            // The person is shown only the corrected reply, and that a correction was made.
            const reply = list.at(-1).reply;
            expect(reply.content).toBe(good);
            expect(reply.steps.map(({ tool, input }) => ({ tool, input: input.slice(0, 27) }))).toEqual([{ tool: 'Drawing check', input: '"Grid" is not a component. ' }]);
            expect(list.some((event) => event.type === 'replace' && event.text === '')).toBe(true);
            expect(list.filter((event) => event.type === 'note').map((event) => event.text)).toEqual(['Correcting what it drew…', '']);
            // Neither the first attempt nor the correction is kept in the chat.
            const kept = (await get(`/chats/${chat.id}`)).data.chat.messages;
            expect(kept.map((message) => message.role)).toEqual(['user', 'assistant']);
            expect(kept[1].content).toBe(good);

            // One chance only: a second attempt that is still wrong is left as written.
            state.chatCalls.length = 0;
            state.replyPlan = [bad, bad];
            const again = await send(chat.id, { content: 'once more' });
            expect(state.chatCalls).toHaveLength(2);
            expect(again.events.at(-1).reply.content).toBe(bad);

            // A second attempt that says nothing is not shown in place of the first.
            state.chatCalls.length = 0;
            state.replyPlan = [bad, ''];
            const empty = await send(chat.id, { content: 'and again' });
            expect(state.chatCalls).toHaveLength(2);
            expect(empty.events.at(-1).reply.content).toBe(bad);

            // A reply with nothing wrong, or with no component at all, is not asked again.
            for (const fine of [good, 'An engine burns fuel.', ['```html', '<div>not for the chat to draw</div>', '```'].join('\n')]) {
                state.chatCalls.length = 0;
                state.replyPlan = [fine];
                await send(chat.id, { content: 'next' });
                expect(state.chatCalls, fine).toHaveLength(1);
            }

            // A chat without the Components skill is not checked.
            const plain = await newChat({ model: 'qwen2.5-coder:7b', skills: [] });
            state.chatCalls.length = 0;
            state.replyPlan = [bad];
            await send(plain.id, { content: 'show the engine' });
            expect(state.chatCalls).toHaveLength(1);
            state.replyPlan = [];

            // A wrong value alone is put right from the fault itself, which lists the values: no page is sent, because a
            // small model given a whole page for one word copied the page's example in place of its own diagram.
            const valued = await newChat({ model: 'qwen2.5-coder:7b' });
            const slip = block('<Flow arrange="horizontal">\n<Node id="a" label="Intake" />\n</Flow>');
            state.chatCalls.length = 0;
            state.replyPlan = [slip, slip.replace('horizontal', 'lr')];
            const fixed = await send(valued.id, { content: 'draw the stroke' });
            expect(state.chatCalls).toHaveLength(2);
            expect(state.chatCalls[1].messages.at(-1).content).toBe('The chat could not draw part of your reply as you meant it:\n- Flow: arrange="horizontal" is not a value it takes. arrange takes: lr, tb, grid, circle.\n\nWrite your whole reply again with only this put right, keeping everything else as you wrote it. Do not mention this correction.');
            expect(fixed.events.at(-1).reply.content).toBe(slip.replace('horizontal', 'lr'));
            expect(componentFaults(slip).valuesOnly).toBe(true);
            expect(componentFaults(block('<Flow arrange="horizontal">\n<Node label="Intake" />\n</Flow>')).valuesOnly).toBe(false);
            expect(componentFaults(good).valuesOnly).toBe(false);
            state.replyPlan = [];

            // The rules themselves: what a page, a drawing or code holds is its own; inner tags must have what they need.
            expect(componentFaults(block('<Frame height="300px">\n<div><h1>Hi</h1></div>\n<script>const a = 1 < 2;</script>\n</Frame>')).faults).toEqual([]);
            expect(componentFaults(block('<Image alt="dot"><svg viewBox="0 0 1 1"><circle r="1" /></svg></Image>')).faults).toEqual([]);
            expect(componentFaults(block('<Flow>\n<Node id="a" label="A" />\n<Edge source="a" target="a" />\n</Flow>')).faults).toEqual([]);
            expect(componentFaults(block('<Flow>\n<Node label="A" />\n<Edge source="a" />\n</Flow>')).faults).toEqual(['Node needs the attribute "id".', 'Edge needs the attribute "target".']);
            expect(componentFaults('<Container layout="flex">\n<div>\n<h2>Parts</h2>\n</div>\n</Container>').faults.map((fault) => fault.split('.')[0])).toEqual(['<Container> is written as plain text, so it shows as text and is not drawn', '"div" is not a component', '"h2" is not a component']);
            // Tags written as plain text are caught, and so is a Button with no label; a tag named in a sentence is not.
            expect(componentFaults('Next:\n\n<ButtonGroup>\n  <Button send="More" />\n</ButtonGroup>').faults).toEqual(expect.arrayContaining([expect.stringMatching(/^<ButtonGroup> is written as plain text/), expect.stringMatching(/^A Button holds its label as text/)]));
            expect(componentFaults('Use the <Button> tag for this.').faults).toEqual([]);
            expect(componentFaults(block('<Model3D src="https://example.com/engine.glb" autoRotate />')).faults).toEqual([]);
            expect(componentFaults(block('<ProgressBar label="Done" />')).faults).toEqual(['ProgressBar needs the attribute "value".']);
            // A value a component does not take is caught, and the fault names the values it does take, as its page lists them.
            expect(componentFaults(block('<Flow arrange="horizontal" background="white">\n<Node id="a" label="A" color="blue" shape="rectangle" />\n<Node id="b" label="B" color="blue" />\n<Edge source="a" target="b" arrow="true" />\n</Flow>')).faults).toEqual([
                'Flow: arrange="horizontal" is not a value it takes. arrange takes: lr, tb, grid, circle.',
                'Flow: background="white" is not a value it takes. background takes: dots, lines, cross, none.',
                'Node: color="blue" is not a value it takes. color takes: primary, secondary, tertiary, success, warning, error, neutral, surface, background.',
                'Edge: arrow="true" is not a value it takes. arrow takes: filled, open, none.'
            ]);
            expect(componentFaults(block('<Flow arrange="tb" background="none" height="400">\n<Node id="a" label="A" color="success" shape="diamond" size="2xl" width="220" />\n<Edge source="a" target="a" type="step" arrow="open" dashed />\n</Flow>')).faults).toEqual([]);
            expect(componentFaults(block('<Flow>\n<Node id="a" width="220px" />\n</Flow>')).faults).toEqual(['Node: width="220px" must be a number with no unit.']);
            expect(componentFaults(block('<Card layout="grid" columns="auto" gap="md" padding="none" shadow bordered="true" backgroundColor="surface">\n<Typography size="4xl" weight="black" color="muted" as="h2">Hi</Typography>\n<Badge color="green">New</Badge>\n</Card>')).faults).toEqual(['Badge: color="green" is not a value it takes. color takes: default, primary, secondary, tertiary, success, warning, error.']);
            // What a page describes in words is not checked, nor is a list the page leaves open.
            expect(componentFaults(block('<Editor mode="code" language="brainfuck">+++</Editor>')).faults).toEqual([]);
            expect(componentFaults(block('<Editor mode="slides">x</Editor>')).faults).toEqual(['Editor: mode="slides" is not a value it takes. mode takes: markdown, document, code.']);
            expect(componentFaults(block('<Container layout="flex" width="50%" columns="3">\n<Button send="Go" size="xs" variant="ghost" width="120px">Go</Button>\n<Icon name="FiCheck" size="2xl" />\n</Container>')).faults).toEqual([]);
            // A block that holds program code or plain words, as some models write, is caught as well.
            expect(componentFaults(block('import { gear } from "kernel.js";\ngear({ teeth: 12 });')).faults).toEqual(['A component block holds only component tags, and one of yours holds something else. Program code goes in an ordinary code fence, and words go outside the block.']);
            expect(componentFaults(block('Here is the panel you asked for.')).faults).toHaveLength(1);
            expect(componentFaults(block('\n  <Badge>Done</Badge>')).faults).toEqual([]);
        });

        test('a reply may use as many tools as it needs, and a call repeated with the same input is not run again', async () => {
            await withKey(async () => {
                // Fourteen rounds, the last with six calls at once: every one runs and tools stay on offer throughout.
                state.toolPlan = [...Array.from({ length: 13 }, (_, index) => [['web_search', { query: `duck model ${index}` }]]), Array.from({ length: 6 }, (_, index) => ['web_search', { query: `goose model ${index}` }])];
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                const { events: list } = await send(chat.id, { content: 'find a duck model' });
                expect(state.searches).toHaveLength(19);
                expect(state.chatCalls).toHaveLength(15);
                expect(state.chatCalls.every((call) => Array.isArray(call.tools))).toBe(true);
                expect(list.at(-1).type).toBe('done');

                // The same search twice: the second is answered from the first, with a note to move on.
                state.searches.length = 0;
                state.chatCalls.length = 0;
                state.toolPlan = [[search], [search]];
                await send(chat.id, { content: 'and again' });
                expect(state.searches).toHaveLength(1);
                expect(state.chatCalls).toHaveLength(3);
                expect(state.chatCalls[2].messages.at(-1)).toMatchObject({ role: 'tool', content: expect.stringContaining('You already called web_search with the same input') });
            });
        });
    });

    describe('Model picker data', () => {
        test('installed models carry a fit badge and whether they are ready now', async () => {
            state.running = [{ name: 'qwen2.5-coder:7b', size: 100, size_vram: 100 }];
            const models = (await get('/models')).data.models;
            const loaded = models.find((model) => model.name === 'qwen2.5-coder:7b');
            const idle = models.find((model) => model.name === 'custom-thing:latest');
            expect(loaded).toMatchObject({ ready: true, canChat: true });
            expect(loaded.fit).toMatchObject({ fits: expect.any(Boolean), label: expect.any(String) });
            expect(idle.ready).toBe(false);
        });
    });

    describe('Files from FilesystemOne', () => {
        test('a model can look through the person\'s files and read one without asking, and each step keeps what came back', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            state.toolPlan = [[['list_my_files', { folder: '/docs' }], ['read_my_file', { path: '/docs/brief.md' }], ['read_my_file', { path: '/docs/missing.md' }]]];
            state.chatCalls = [];
            const { events } = await send(chat.id, { content: 'what does my brief say?' });
            state.toolPlan = [];
            expect(events.some((event) => event.type === 'permission')).toBe(false);
            const told = state.chatCalls[1].messages.filter((message) => message.role === 'tool').map((message) => message.content);
            expect(told[0]).toContain('file /docs/brief.md (1 KB)');
            expect(told[0]).toContain('folder /docs');
            expect(told[1]).toBe('brief.md:\n\nThe client wants a harbour survey by June.');
            expect(told[2]).toMatch(/^Nothing was read: /);
            expect(events.filter((event) => event.type === 'toolAnswer').map((event) => event.answer)).toEqual(told);
            const stored = (await Chat.findById(chat.id)).messages.at(-1).steps;
            expect(stored.map((step) => [step.tool, step.input, step.failed])).toEqual([['List files', '/docs', null], ['Read a file', '/docs/brief.md', null], ['Read a file', '/docs/missing.md', 'failed']]);
            expect(stored[1].answer).toBe(told[1]);
        });

        test('browse passes the caller’s own sign-in through', async () => {
            const response = await get('/files/browse?path=/');
            expect(response.status).toBe(200);
            expect(response.data).toMatchObject({ path: '/', parent: null });
            expect(response.data.entries).toEqual([
                { name: 'docs', path: '/docs', type: 'folder', size: 0, updatedAt: null },
                { name: 'brief.md', path: '/docs/brief.md', type: 'file', size: 42, updatedAt: null }
            ]);

            const mine = client.cookieJar.getCookieStringSync(client.baseURL).match(/accessToken=([^;]+)/)[1];
            expect(filesCalls[0].url).toMatch(/^http:\/\/fs-one\.test\/api\/v1\/files\/directory\/contents\?filePath=%2F/);
            expect(filesCalls[0].cookie).toContain(`accessToken=${mine}`);

            await testStartup.loginAsUser('owner');
            await get('/files/browse?path=/docs');
            const theirs = client.cookieJar.getCookieStringSync(client.baseURL).match(/accessToken=([^;]+)/)[1];
            expect(theirs).not.toBe(mine);
            expect(filesCalls[1].cookie).toContain(`accessToken=${theirs}`);
        });

        test('needs sign-in', async () => {
            client.clearCookies();
            client.authenticatedAs = null;
            expect((await get('/files/browse?path=/')).status).toBe(401);
            expect(filesCalls).toHaveLength(0);
        });

        test('a folder FilesystemOne refuses stays refused', async () => {
            const response = await get('/files/browse?path=/private');
            expect(response.status).toBe(404);
            expect(response.data).toMatchObject({ code: 'FILE_NOT_FOUND' });
        });

        test('a file FilesystemOne refuses stays refused', async () => {
            const response = await get('/files/text?path=/private/secret.md');
            expect(response.status).toBe(404);
            expect(response.data).toMatchObject({ code: 'FILE_NOT_FOUND' });
            const chat = await newChat();
            expect((await post(`/chats/${chat.id}/attachments`, { path: '/private/secret.md' })).status).toBe(404);
        });

        test('reads a text file with its size against the model', async () => {
            const response = await get('/files/text?path=/docs/brief.md&model=qwen2.5-coder:7b');
            expect(response.data).toMatchObject({
                name: 'brief.md', path: '/docs/brief.md', text: 'The client wants a harbour survey by June.', capacityTokens: 32768, fits: true
            });
        });

        test('attaches a text file and a PDF from FilesystemOne as copies', async () => {
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            expect((await post(`/chats/${chat.id}/attachments`, { path: '/docs/brief.md' })).status).toBe(201);
            const second = await post(`/chats/${chat.id}/attachments`, { path: '/docs/report.pdf' });
            expect(second.data.chat.pendingAttachments.map((file) => [file.name, file.source, file.path])).toEqual([
                ['brief.md', 'fs-one', '/docs/brief.md'],
                ['report.pdf', 'fs-one', '/docs/report.pdf']
            ]);

            fsFiles['/docs/brief.md'].content = 'CHANGED AFTERWARDS';
            await send(chat.id, { content: 'Compare them.' });
            const sent = state.chatCalls[0].messages[0].content;
            expect(sent).toContain('harbour survey by June');
            expect(sent).toContain('Dredging costs rose in March');
            expect(sent).not.toContain('CHANGED AFTERWARDS');
            fsFiles['/docs/brief.md'].content = 'The client wants a harbour survey by June.';
        });

        test('an image is refused in plain words', async () => {
            const response = await get('/files/text?path=/docs/photo.png');
            expect(response.status).toBe(415);
            expect(response.data).toMatchObject({ code: 'FILE_UNREADABLE' });
        });

        test('an oversized file is refused', async () => {
            fsFiles['/docs/huge.txt'] = { type: 'binary', bytes: Buffer.alloc(MAX_ATTACHMENT_BYTES + 10, 'a') };
            const response = await get('/files/text?path=/docs/huge.txt');
            delete fsFiles['/docs/huge.txt'];
            expect(response.status).toBe(413);
            expect(response.data).toMatchObject({ code: 'FILE_TOO_LARGE' });
        }, 30000);

        test('a FilesystemOne that is waking up is asked again until it answers', async () => {
            setFilesWakeMs(5000, 10);
            try {
                state.filesWaking = 2;
                const response = await get('/files/browse?path=/');
                expect(response.status).toBe(200);
                expect(response.data.entries.length).toBeGreaterThan(0);
                expect(state.filesWaking).toBe(0);
            } finally {
                setFilesWakeMs();
                state.filesWaking = 0;
            }
        });

        test('FilesystemOne being down gives a plain message and chat still works', async () => {
            state.filesDown = true;
            setFilesWakeMs(0);
            for (const response of [await get('/files/browse?path=/'), await get('/files/text?path=/docs/brief.md')]) {
                expect(response.status).toBe(503);
                expect(response.data).toMatchObject({ code: 'FILES_DOWN' });
                expect(response.data.message).toMatch(/FilesystemOne may still be starting up/);
                expect(response.data.action).toMatch(/Try again in a minute/);
            }
            setFilesWakeMs();
            const chat = await newChat({ model: 'qwen2.5-coder:7b' });
            expect((await send(chat.id, { content: 'still here?' })).response.status).toBe(200);
        });

        test('saves text into a folder with the caller’s sign-in', async () => {
            const response = await post('/files/save', { folder: '/docs', name: 'chat.md', content: '# Chat' });
            expect(response.status).toBe(201);
            expect(response.data.path).toBe('/docs/chat.md');
            const call = filesCalls[0];
            expect(call.method).toBe('POST');
            expect(JSON.parse(call.body)).toEqual({ filePath: '/docs/chat.md', content: '# Chat', onConflict: 'rename' });
            expect(call.cookie).toContain(`csrfToken=${call.csrf}`);
        });

        test('a file a reply showed from the web is fetched by the server and saved into a folder, and a model can save one too once allowed', async () => {
            const uploads = [];
            const fetched = [];
            const before = setRemoteTransport({
                lookup: async () => [{ address: '93.184.216.34', family: 4 }],
                request: async (url) => {
                    fetched.push(url.href);
                    return { status: 200, headers: { 'content-type': 'image/png' }, stream: Readable.from([Buffer.from('png-bytes')]) };
                }
            });
            setFilesFetch(async (url, options) => {
                if (!String(url).endsWith('/files/upload')) return fakeFilesystemOne(url, options);
                uploads.push({ folder: options.body.get('basePath'), paths: options.body.get('relativePaths'), conflict: options.body.get('onConflict'), text: await options.body.get('files').text() });
                return new Response(JSON.stringify({ success: true }), { status: 201, headers: { 'Content-Type': 'application/json' } });
            });
            try {
                // From the Save button: the address is fetched here, with the usual checks, and uploaded without overwriting.
                const response = await post('/files/save-address', { folder: '/docs', name: 'duck.png', url: 'https://models.example/duck.png' });
                expect(response.status).toBe(201);
                expect(response.data.path).toBe('/docs/duck.png');
                expect(fetched).toEqual(['https://models.example/duck.png']);
                expect(uploads).toEqual([{ folder: '/docs', paths: '["duck.png"]', conflict: 'rename', text: 'png-bytes' }]);
                // Only https addresses, and only a plain file name.
                expect((await post('/files/save-address', { folder: '/docs', name: 'duck.png', url: 'http://models.example/duck.png' })).status).toBe(400);
                expect((await post('/files/save-address', { folder: '/docs', name: '../duck.png', url: 'https://models.example/duck.png' })).status).toBe(400);

                // From a model: it asks first, then saves what it found.
                const chat = await newChat({ model: 'qwen2.5-coder:7b' });
                state.toolPlan = [[['save_to_files', { folder: '/docs', name: 'found.png', address: 'https://models.example/found.png' }]]];
                state.chatCalls = [];
                const pending = send(chat.id, { content: 'save that picture' });
                for (let tries = 0; tries < 200 && !waitingPermissions().some((waiting) => waiting.chatId === chat.id); tries += 1) await wait(25);
                const waiting = waitingPermissions().find((one) => one.chatId === chat.id);
                expect(waiting).toBeTruthy();
                expect((await post(`/chats/${chat.id}/permission`, { id: waiting.id, answer: 'once' })).status).toBe(200);
                const { events } = await pending;
                // A folder a model names goes inside the person's own folder, where they can see it, and the model is
                // told the full path.
                const home = uploads.at(-1).folder.replace(/\/docs$/, '');
                expect(home).toMatch(/^\/[a-z0-9]+$/);
                expect(events.find((event) => event.type === 'tool')).toMatchObject({ tool: 'Save to files', input: `${home}/docs/found.png` });
                expect(uploads.at(-1)).toMatchObject({ folder: `${home}/docs`, paths: '["found.png"]' });
                expect(state.chatCalls[1].messages.at(-1).content).toContain(`to ${home}/docs/found.png.`);

                // A folder already in their home, or one climbing out of it, stays inside it.
                for (const [folder, where] of [[`${home}/Gearbox/`, `${home}/Gearbox`], ['/../Gearbox', `${home}/Gearbox`], ['/', home]]) {
                    state.toolPlan = [[['save_to_files', { folder, name: 'found.png', address: 'https://models.example/found.png' }]]];
                    const again = send(chat.id, { content: 'save it again' });
                    for (let tries = 0; tries < 200 && !waitingPermissions().some((one) => one.chatId === chat.id); tries += 1) await wait(25);
                    await post(`/chats/${chat.id}/permission`, { id: waitingPermissions().find((one) => one.chatId === chat.id).id, answer: 'once' });
                    await again;
                    expect(uploads.at(-1).folder).toBe(where);
                }
            } finally {
                setRemoteTransport(before);
                setFilesFetch(fakeFilesystemOne);
                state.toolPlan = [];
            }
        });

        test('a folder that cannot be written to is said plainly', async () => {
            const response = await post('/files/save', { folder: '/locked', name: 'chat.md', content: '# Chat' });
            expect(response.status).toBe(403);
            expect(response.data).toMatchObject({ code: 'FILE_NOT_SAVED' });
        });

        test('rejects a path that is not absolute', async () => {
            expect((await get('/files/browse?path=docs')).status).toBe(400);
            expect((await get('/files/text?path=/docs//brief.md')).status).toBe(400);
            expect(filesCalls).toHaveLength(0);
        });
    });
});

describe('Files loaded from the web', () => {
    const get = (path) => client.get(`/api/v1${path}`);
    const address = (url) => `/files/remote?url=${encodeURIComponent(url)}`;
    const body = (text, headers = {}) => ({ status: 200, headers, stream: Readable.from([Buffer.from(text)]) });
    let requests = [];
    let pages = {};

    beforeAll(() => {
        setRemoteTransport({
            lookup: async (host) => {
                if (host === 'intranet.example') return [{ address: '10.0.0.5', family: 4 }];
                if (host === 'mixed.example') return [{ address: '93.184.216.34', family: 4 }, { address: '127.0.0.1', family: 4 }];
                if (host === 'gone.example') throw new Error('ENOTFOUND');
                return [{ address: '93.184.216.34', family: 4 }];
            },
            request: async (url, pinned) => {
                requests.push({ url: url.href, pinned: pinned.address });
                const page = pages[url.href];
                if (!page) return { status: 404, headers: {}, stream: Readable.from([]) };
                return typeof page === 'function' ? page() : page;
            }
        });
    });

    afterAll(() => setRemoteTransport(null));

    beforeEach(async () => {
        requests = [];
        pages = {};
        await testStartup.loginAsUser('user');
    });

    test.each([
        ['8.8.8.8', true], ['93.184.216.34', true], ['2606:4700:4700::1111', true], ['192.0.66.161', true], ['198.51.99.1', true],
        ['192.0.0.8', false], ['192.0.2.1', false], ['198.51.100.7', false], ['203.0.113.9', false],
        ['127.0.0.1', false], ['10.1.2.3', false], ['172.16.0.1', false], ['172.31.255.255', false], ['192.168.1.10', false],
        ['169.254.169.254', false], ['100.64.0.1', false], ['0.0.0.0', false], ['224.0.0.1', false],
        ['::1', false], ['fd00::1', false], ['fe80::1', false], ['::ffff:127.0.0.1', false], ['not-an-ip', false]
    ])('%s is public: %s', (ip, expected) => {
        expect(isPublicAddress(ip)).toBe(expected);
    });

    test('needs sign-in', async () => {
        client.clearCookies();
        client.authenticatedAs = null;
        expect((await get(address('https://files.example/a.obj'))).status).toBe(401);
        expect(requests).toHaveLength(0);
    });

    test('passes a public https file through', async () => {
        pages['https://files.example/house.obj'] = body('v 0 0 0\nv 1 0 0\n', { 'content-type': 'text/html' });
        const response = await get(address('https://files.example/house.obj'));
        expect(response.status).toBe(200);
        expect(String(response.data)).toContain('v 1 0 0');
        expect(response.headers['content-type']).toMatch(/octet-stream/);
        expect(response.headers['content-disposition']).toMatch(/attachment/);
        expect(response.headers['x-content-type-options']).toBe('nosniff');
        expect(requests).toEqual([{ url: 'https://files.example/house.obj', pinned: '93.184.216.34' }]);
    });

    test.each([
        'http://files.example/a.obj',
        'https://user:pass@files.example/a.obj',
        'https://127.0.0.1/a.obj',
        'https://[::1]/a.obj',
        'https://169.254.169.254/latest/meta-data',
        'https://intranet.example/a.obj',
        'https://mixed.example/a.obj'
    ])('refuses %s without contacting it', async (url) => {
        const response = await get(address(url));
        expect([400, 403]).toContain(response.status);
        expect(requests).toHaveLength(0);
    });

    test('says plainly when an address is blocked', async () => {
        const response = await get(address('https://intranet.example/a.obj'));
        expect(response.data).toMatchObject({ success: false, code: 'REMOTE_BLOCKED' });
        expect(response.data.action).toMatch(/public https/);
    });

    test('follows a redirect to a public address but not to a private one', async () => {
        pages['https://files.example/old.pdf'] = { status: 302, headers: { location: 'https://files.example/new.pdf' }, stream: Readable.from([]) };
        pages['https://files.example/new.pdf'] = body('%PDF-1.4');
        expect(String((await get(address('https://files.example/old.pdf'))).data)).toContain('%PDF-1.4');

        pages['https://files.example/trap.pdf'] = { status: 302, headers: { location: 'https://intranet.example/secret.pdf' }, stream: Readable.from([]) };
        const trapped = await get(address('https://files.example/trap.pdf'));
        expect(trapped.data).toMatchObject({ code: 'REMOTE_BLOCKED' });
        expect(requests.some((entry) => entry.url.includes('intranet'))).toBe(false);
    });

    test('stops after too many redirects', async () => {
        pages['https://files.example/loop'] = () => ({ status: 302, headers: { location: 'https://files.example/loop' }, stream: Readable.from([]) });
        const response = await get(address('https://files.example/loop'));
        expect(response.status).toBe(502);
        expect(response.data).toMatchObject({ code: 'REMOTE_FAILED' });
    });

    test('a site that is limiting requests is tried once more, then reported as busy', async () => {
        let calls = 0;
        pages['https://files.example/slow.png'] = () => (++calls === 1 ? { status: 429, headers: {}, stream: Readable.from([]) } : body('picture'));
        const first = await get(address('https://files.example/slow.png'));
        expect(first.status).toBe(200);
        expect(calls).toBe(2);

        pages['https://files.example/busy.png'] = () => ({ status: 429, headers: {}, stream: Readable.from([]) });
        const second = await get(address('https://files.example/busy.png'));
        expect(second.status).toBe(503);
        expect(second.data).toMatchObject({ code: 'REMOTE_BUSY' });
    });

    test('a missing file, an unknown host and an oversized file each get a plain message', async () => {
        const missing = await get(address('https://files.example/missing.glb'));
        expect(missing.status).toBe(404);
        expect(missing.data).toMatchObject({ code: 'REMOTE_NOT_FOUND', message: 'Nothing exists at this address.' });
        expect((await get(address('https://gone.example/a.glb'))).data).toMatchObject({ code: 'REMOTE_FAILED' });
        pages['https://files.example/huge.mp4'] = body('x', { 'content-length': String(MAX_REMOTE_BYTES + 1) });
        const huge = await get(address('https://files.example/huge.mp4'));
        expect(huge.status).toBe(413);
        expect(huge.data).toMatchObject({ code: 'REMOTE_TOO_LARGE' });
    });

    test('needs a url', async () => {
        expect((await get('/files/remote')).status).toBe(400);
        expect((await get(address('not a url'))).status).toBe(400);
    });
});

describe('Linked devices', () => {
    const TAGS = { models: [{ name: 'tiny:1b', model: 'tiny:1b', size: 1000, digest: 'abc', details: { parameter_size: '1B', quantization_level: 'Q4' }, modified_at: new Date().toISOString() }] };
    const through = (deviceId) => ({ headers: { 'X-Mollama-Device': deviceId }, validateStatus: () => true });
    const hardware = { hostname: 'laptop', platform: 'darwin', cpu: 'Test CPU', cpuThreads: 8, memoryMb: 16384, gpus: [], gpuMemoryMb: 0, diskFreeMb: 50000, diskTotalMb: 100000 };
    const open = [];

    // Stands in for the desktop app: connects, says hello, and answers what the server asks of it.
    const fakeApp = (deviceId, key, { holdPull = false, network = null, build = 'b12591' } = {}) => new Promise((resolve) => {
        const ws = new WebSocket(`ws://localhost:${testStartup.port}/devices`);
        const held = [];
        const app = { ws, welcomed: false, requests: [], orders: [], probes: [], runs: [], pooled: [], poolDown: false, reach: { reachable: true }, closed: new Promise((done) => ws.addEventListener('close', done)), release: () => held.splice(0).forEach((answer) => answer()) };
        open.push(ws);
        const send = (message) => ws.send(JSON.stringify(message));
        ws.addEventListener('open', () => send({ type: 'hello', deviceId, key }));
        ws.addEventListener('close', () => resolve(app));
        ws.addEventListener('message', (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'welcome') {
                app.welcomed = true;
                return resolve(app);
            }
            // Pooling as the desktop app does it: an order to lend with an address to admit opens the gate on 'bind'.
            if (message.type === 'pool' && message.action === 'apply') {
                app.orders.push({ role: message.role, bind: message.bind ?? null, allow: message.allow ?? [] });
                const lending = message.role === 'lend' && Boolean(message.bind) && (message.allow || []).length > 0;
                return send({ type: 'pool-state', data: { build, role: message.role, setup: 'ready', progress: null, lending, address: lending ? message.bind : null, allow: lending ? message.allow : [], port: 50052, detail: null } });
            }
            if (message.type === 'pool' && message.action === 'run') {
                app.runs.push({ model: message.model, helpers: message.helpers, split: message.split, ctx: message.ctx });
                send({ type: 'pool-state', data: { build, role: 'main', setup: 'ready', lending: false, address: null, allow: [], port: 50052, running: message.model, detail: null } });
                return send({ id: message.id, type: 'answer', data: { ready: true } });
            }
            if (message.type === 'request' && message.via === 'pool') {
                app.pooled.push({ path: message.path, body: JSON.parse(message.body || '{}') });
                if (app.poolDown) return send({ id: message.id, type: 'error', code: 'POOL_DOWN' });
                send({ id: message.id, type: 'head', status: 200, headers: { 'content-type': 'text/event-stream' } });
                for (const piece of ['Pooled ', 'hello']) send({ id: message.id, type: 'chunk', data: Buffer.from(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: piece } }] })}\n\n`).toString('base64') });
                send({ id: message.id, type: 'chunk', data: Buffer.from(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 5, completion_tokens: 2 } })}\n\ndata: [DONE]\n\n`).toString('base64') });
                return send({ id: message.id, type: 'end' });
            }
            if (message.type === 'pool' && message.action === 'probe') {
                app.probes.push(`${message.address}:${message.port}`);
                return send({ id: message.id, type: 'answer', data: app.reach });
            }
            if (message.type === 'ask') {
                return send({ id: message.id, type: 'answer', data: message.what === 'disk' ? { diskFreeMb: 4242, diskTotalMb: 9000 } : { ...hardware, memoryMb: 32768, ...((app.network || network) ? { network: app.network || network } : {}) } });
            }
            if (message.type !== 'request') return undefined;
            app.requests.push(`${message.method} ${message.path}`);
            const body = message.path === '/api/version' ? { version: '9.9.9' } : message.path === '/api/tags' ? TAGS : message.path === '/api/ps' ? { models: [] }
                : message.path === '/api/show' ? { capabilities: ['completion', 'tools'], model_info: { 'llama.context_length': 8192 }, details: {} } : null;
            if (!body) {
                const notFound = () => {
                    send({ id: message.id, type: 'head', status: 404, headers: { 'content-type': 'application/json' } });
                    send({ id: message.id, type: 'chunk', data: Buffer.from('{"error":"not found"}').toString('base64') });
                    send({ id: message.id, type: 'end' });
                };
                if (holdPull && message.path === '/api/pull') return held.push(notFound);
                return notFound();
            }
            const text = JSON.stringify(body);
            send({ id: message.id, type: 'head', status: 200, headers: { 'content-type': 'application/json' } });
            send({ id: message.id, type: 'chunk', data: Buffer.from(text.slice(0, 5)).toString('base64') });
            send({ id: message.id, type: 'chunk', data: Buffer.from(text.slice(5)).toString('base64') });
            return send({ id: message.id, type: 'end' });
        });
    });

    const link = async (hostname = 'laptop') => {
        const response = await client.post('/api/v1/devices/link', { hostname, hardware });
        expect(response.status).toBe(201);
        return { id: response.data.device.id, key: response.data.key };
    };
    const listed = async () => (await client.get('/api/v1/devices')).data.devices;
    test('which computers can pool is worked out from their private addresses: same home network or a direct cable, never a Public network', () => {
        expect(subnetOf('10.0.0.135', 24)).toBe('10.0.0.0/24');
        expect(subnetOf('172.26.140.9', 20)).toBe('172.26.128.0/20');
        expect(subnetOf('169.254.40.2', 16)).toBe('169.254.0.0/16');
        expect(subnetOf('not an address', 24)).toBeNull();
        expect(['10.1.2.3', '172.16.0.1', '172.32.0.1', '192.168.1.5', '169.254.9.9', '8.8.8.8'].map(privateKind)).toEqual(['home', 'home', null, 'home', 'cable', null]);
        const pc = (id, publicAddress, addresses, more = {}) => ({ id, name: id, online: true, paused: false, publicAddress, hardware: { memoryMb: 32768, gpuMemoryMb: 8192, network: { addresses } }, ...more });
        const { groups, left } = poolGroups([
            pc('pc', '1.1.1.1', [{ address: '10.0.0.135', prefix: 24, wired: false }, { address: '169.254.1.1', prefix: 16, wired: true }]),
            pc('mac', '1.1.1.1', [{ address: '10.0.0.66', prefix: 24, wired: false }, { address: '169.254.1.2', prefix: 16, wired: true }]),
            pc('away', '2.2.2.2', [{ address: '10.0.0.70', prefix: 24 }]),
            pc('cafe', '1.1.1.1', [{ address: '10.0.0.80', prefix: 24, category: 'public' }]),
            pc('asleep', '1.1.1.1', [{ address: '10.0.0.90', prefix: 24 }], { online: false }),
            pc('resting', '1.1.1.1', [{ address: '10.0.0.91', prefix: 24 }], { paused: true }),
            { id: 'old', name: 'old', online: true, publicAddress: '1.1.1.1', hardware: { memoryMb: 8192 } }
        ]);
        // A direct cable comes first, then the home network; both name the same two computers.
        expect(groups.map((group) => [group.via, group.network, group.computers.map((one) => one.id)])).toEqual([['cable', '169.254.0.0/16', ['pc', 'mac']], ['home', '10.0.0.0/24', ['pc', 'mac']]]);
        expect(groups[0]).toMatchObject({ wired: true, combined: { gpuMemoryMb: 16384, memoryMb: 65536 } });
        // Another home with the same private addresses is not the same network.
        expect(Object.fromEntries(left.map((one) => [one.id, one.reason]))).toEqual({ away: 'alone', cafe: 'public', asleep: 'offline', resting: 'paused', old: 'update' });
        expect(left.find((one) => one.id === 'cafe').message).toMatch(/Public/);
    });

    test('a person sees which of their linked computers can pool, read from what each desktop app reports when it connects', async () => {
        await testStartup.loginAsUser('admin');
        const home = (address, extra = {}) => ({ addresses: [{ address, prefix: 24, interface: 'Wi-Fi', wired: false, ...extra }] });
        const pc = await link('pool-pc');
        const mac = await link('pool-mac');
        const cafe = await link('pool-cafe');
        const old = await link('pool-old');
        const asleep = await link('pool-asleep');
        await fakeApp(pc.id, pc.key, { network: home('10.9.0.10') });
        await fakeApp(mac.id, mac.key, { network: home('10.9.0.20') });
        await fakeApp(cafe.id, cafe.key, { network: home('10.9.0.30', { category: 'public' }) });
        await fakeApp(old.id, old.key);
        let pools = null;
        for (let tries = 0; tries < 60; tries += 1) {
            pools = (await client.get('/api/v1/devices/pools')).data;
            if (pools.groups.some((group) => group.network === '10.9.0.0/24' && group.computers.length === 2)) break;
            await new Promise((resolve) => setTimeout(resolve, 50));
        }
        const group = pools.groups.find((one) => one.network === '10.9.0.0/24');
        expect(group.computers.map((one) => one.address).sort()).toEqual(['10.9.0.10', '10.9.0.20']);
        expect(group).toMatchObject({ via: 'home', wired: false });
        expect(group.combined.memoryMb).toBe(65536);
        const reason = (id) => pools.left.find((one) => one.id === id)?.reason;
        expect([reason(cafe.id), reason(old.id), reason(asleep.id)]).toEqual(['public', 'update', 'offline']);
        // The public address each link came from is never sent to the page.
        expect(JSON.stringify(pools)).not.toMatch(/publicAddress|127\.0\.0\.1|::1/);

        // Another person sees none of these computers.
        await testStartup.loginAsUser('user');
        const theirs = (await client.get('/api/v1/devices/pools')).data;
        expect(JSON.stringify(theirs)).not.toContain(pc.id);
        await testStartup.loginAsUser('admin');
    });

    test('a lending computer is told to admit only its main computer, and the main computer checks that it can reach it', async () => {
        await testStartup.loginAsUser('admin');
        const home = (address) => ({ addresses: [{ address, prefix: 24, interface: 'Ethernet', wired: true }] });
        const until = async (check) => {
            let last = null;
            for (let tries = 0; tries < 80; tries += 1) {
                last = (await client.get('/api/v1/devices/pools')).data;
                if (check(last)) return last;
                await new Promise((resolve) => setTimeout(resolve, 50));
            }
            return last;
        };
        const ordered = async (app, want) => {
            for (let tries = 0; tries < 60 && JSON.stringify(app.orders.at(-1)) !== JSON.stringify(want); tries += 1) await new Promise((resolve) => setTimeout(resolve, 50));
            return app.orders.at(-1);
        };
        const pc = await link('lend-pc');
        const mac = await link('lend-mac');
        const pcApp = await fakeApp(pc.id, pc.key, { network: home('10.9.1.10') });
        const macApp = await fakeApp(mac.id, mac.key, { network: home('10.9.1.20') });
        const groupOf = (pools) => pools.groups.find((group) => group.network === '10.9.1.0/24');
        const partOf = (pools, id) => groupOf(pools)?.computers.find((one) => one.id === id);
        await until((pools) => groupOf(pools)?.computers.length === 2);
        // Computers that never pooled are told nothing.
        expect([pcApp.orders, macApp.orders]).toEqual([[], []]);

        // Lending with no main computer yet sets the computer up but keeps its gate shut.
        expect((await client.put(`/api/v1/devices/${mac.id}/pool`, { role: 'lend' })).status).toBe(200);
        expect(await ordered(macApp, { role: 'lend', bind: null, allow: [] })).toEqual({ role: 'lend', bind: null, allow: [] });

        // Once there is a main computer on the same network, the lender admits that one address and nothing else.
        const set = await client.put(`/api/v1/devices/${pc.id}/pool`, { role: 'main' });
        expect(set.data.main).toBe(pc.id);
        expect(await ordered(pcApp, { role: 'main', bind: null, allow: [] })).toEqual({ role: 'main', bind: null, allow: [] });
        expect(await ordered(macApp, { role: 'lend', bind: '10.9.1.20', allow: ['10.9.1.10'] })).toEqual({ role: 'lend', bind: '10.9.1.20', allow: ['10.9.1.10'] });
        let pools = await until((found) => partOf(found, mac.id)?.reachable === true);
        expect(pcApp.probes).toContain('10.9.1.20:50052');
        expect(partOf(pools, mac.id)).toMatchObject({ role: 'lend', setup: 'ready', lending: true, reachable: true, detail: null });
        expect(partOf(pools, pc.id)).toMatchObject({ role: 'main', lending: false });

        // A gate that turns the main computer away says so.
        pcApp.reach = { reachable: false, detail: 'It turned this computer away.' };
        await client.put(`/api/v1/devices/${mac.id}/pool`, { role: 'off' });
        await client.put(`/api/v1/devices/${mac.id}/pool`, { role: 'lend' });
        pools = await until((found) => partOf(found, mac.id)?.reachable === false);
        expect(partOf(pools, mac.id).detail).toBe('It turned this computer away.');

        // Another person cannot change these computers, and a part that does not exist is refused.
        await testStartup.loginAsUser('user');
        expect((await client.put(`/api/v1/devices/${mac.id}/pool`, { role: 'off' })).status).toBe(404);
        await testStartup.loginAsUser('admin');
        expect((await client.put(`/api/v1/devices/${mac.id}/pool`, { role: 'boss' })).status).toBe(400);

        // A person has one main computer: choosing another sets the first one off, and lending stops.
        await client.put(`/api/v1/devices/${mac.id}/pool`, { role: 'main' });
        expect(await ordered(pcApp, { role: 'off', bind: null, allow: [] })).toEqual({ role: 'off', bind: null, allow: [] });
        pools = await until((found) => found.main === mac.id && partOf(found, pc.id)?.role === 'off');
        expect(partOf(pools, mac.id)).toMatchObject({ role: 'main', lending: false });

        // When the main computer goes offline, the lender shuts its gate.
        await client.put(`/api/v1/devices/${pc.id}/pool`, { role: 'lend' });
        expect(await ordered(pcApp, { role: 'lend', bind: '10.9.1.10', allow: ['10.9.1.20'] })).toEqual({ role: 'lend', bind: '10.9.1.10', allow: ['10.9.1.20'] });
        macApp.ws.close();
        await macApp.closed;
        for (let tries = 0; tries < 60 && pcApp.orders.at(-1).allow.length > 0; tries += 1) await new Promise((resolve) => setTimeout(resolve, 50));
        expect(await ordered(pcApp, { role: 'lend', bind: null, allow: [] })).toEqual({ role: 'lend', bind: null, allow: [] });
        await client.put(`/api/v1/devices/${pc.id}/pool`, { role: 'off' });
        await client.put(`/api/v1/devices/${mac.id}/pool`, { role: 'off' });
    });

    test('a lending computer on another llama.cpp build than the main computer is told why it cannot pool', async () => {
        await testStartup.loginAsUser('admin');
        const home = (address) => ({ addresses: [{ address, prefix: 24, interface: 'Wi-Fi', wired: false }] });
        const main = await link('build-main');
        const helper = await link('build-helper');
        await fakeApp(main.id, main.key, { network: home('10.9.2.10') });
        await fakeApp(helper.id, helper.key, { network: home('10.9.2.20'), build: 'b99999' });
        await client.put(`/api/v1/devices/${main.id}/pool`, { role: 'main' });
        await client.put(`/api/v1/devices/${helper.id}/pool`, { role: 'lend' });
        let part = null;
        for (let tries = 0; tries < 60 && !part?.detail; tries += 1) {
            const pools = (await client.get('/api/v1/devices/pools')).data;
            part = pools.groups.find((group) => group.network === '10.9.2.0/24')?.computers.find((one) => one.id === helper.id);
            if (!part?.detail) await new Promise((resolve) => setTimeout(resolve, 50));
        }
        expect(part.detail).toMatch(/pools a different way/);
        await client.put(`/api/v1/devices/${helper.id}/pool`, { role: 'off' });
        await client.put(`/api/v1/devices/${main.id}/pool`, { role: 'off' });
    });

    test('a ready pool is offered as a computer, starts the model on the main computer and answers through it', async () => {
        await testStartup.loginAsUser('admin');
        const home = (address) => ({ addresses: [{ address, prefix: 24, interface: 'Ethernet', wired: true }] });
        const main = await link('run-main');
        const lender = await link('run-lender');
        const mainApp = await fakeApp(main.id, main.key, { network: home('10.9.4.10') });
        await fakeApp(lender.id, lender.key, { network: home('10.9.4.20') });
        const pickable = async () => (await client.get('/api/v1/devices?with=online')).data.devices.find((one) => one.kind === 'pool');
        expect(await pickable()).toBeUndefined();

        await client.put(`/api/v1/devices/${main.id}/pool`, { role: 'main' });
        await client.put(`/api/v1/devices/${lender.id}/pool`, { role: 'lend' });
        let pool = null;
        for (let tries = 0; tries < 80 && !pool; tries += 1) {
            pool = await pickable();
            if (!pool) await new Promise((resolve) => setTimeout(resolve, 50));
        }
        expect(pool).toMatchObject({ name: 'Pool: run-main + run-lender', online: true, ollama: { available: true } });
        expect(pool.hardware.memoryMb).toBe(65536);

        // The pool's models are the main computer's.
        const models = (await client.client.get('/api/v1/models', through(pool.id))).data.models.map((one) => one.name);
        expect(models).toEqual(['tiny:1b']);

        const made = await client.post('/api/v1/chats', { model: 'tiny:1b', device: pool.id });
        const sent = await client.client.post(`/api/v1/chats/${made.data.chat.id}/messages`, { content: 'Hello pool' }, { responseType: 'text', validateStatus: () => true });
        const events = String(sent.data).trim().split('\n').map((line) => JSON.parse(line));
        expect(events.filter((event) => event.type === 'token').map((event) => event.text).join('')).toContain('Pooled hello');
        // Split by what the model needs (tiny here), on the fastest device first.
        expect(mainApp.runs).toEqual([{ model: 'tiny:1b', helpers: ['10.9.4.20:50052'], split: [0.8], ctx: 8192 }]);
        expect(mainApp.pooled[0]).toMatchObject({ path: '/v1/chat/completions', body: { model: 'tiny:1b', stream: true } });
        expect(JSON.stringify(mainApp.pooled[0].body.messages)).toContain('Hello pool');

        // A second message uses the model already running: it is not started again.
        await client.client.post(`/api/v1/chats/${made.data.chat.id}/messages`, { content: 'Again' }, { responseType: 'text', validateStatus: () => true });
        expect(mainApp.runs).toHaveLength(1);

        // A computer that drops out ends the reply with a plain reason.
        mainApp.poolDown = true;
        const dropped = await client.client.post(`/api/v1/chats/${made.data.chat.id}/messages`, { content: 'Still there?' }, { responseType: 'text', validateStatus: () => true });
        const failed = String(dropped.data).trim().split('\n').map((line) => JSON.parse(line)).find((event) => event.type === 'error');
        expect(failed).toMatchObject({ code: 'POOL_STOPPED' });
        expect(failed.message).toMatch(/stopped answering/);

        // With no computer lending, the pool is not offered and a chat on it says why.
        await client.put(`/api/v1/devices/${lender.id}/pool`, { role: 'off' });
        expect(await pickable()).toBeUndefined();
        const notReady = await client.client.post(`/api/v1/chats/${made.data.chat.id}/messages`, { content: 'Anyone?' }, { responseType: 'text', validateStatus: () => true });
        expect(String(notReady.data)).toMatch(/POOL_NOT_READY|pool is not ready/);

        // Another person cannot use this pool.
        await testStartup.loginAsUser('user');
        const theirs = await client.client.post('/api/v1/chats', { model: 'tiny:1b', device: pool.id }, { validateStatus: () => true });
        expect(theirs.status).toBeGreaterThanOrEqual(400);
        await testStartup.loginAsUser('admin');
        await client.put(`/api/v1/devices/${main.id}/pool`, { role: 'off' });
    });

    test('a network check on demand reads each computer again, so one that moved network is seen where it is now', async () => {
        await testStartup.loginAsUser('admin');
        const home = (address) => ({ addresses: [{ address, prefix: 24, interface: 'Wi-Fi', wired: false }] });
        const one = await link('check-one');
        const two = await link('check-two');
        await fakeApp(one.id, one.key, { network: home('10.9.5.10') });
        const twoApp = await fakeApp(two.id, two.key, { network: home('10.9.6.20') });
        const sharing = (pools) => pools.groups.some((group) => group.computers.some((c) => c.id === one.id) && group.computers.some((c) => c.id === two.id));
        let pools = (await client.get('/api/v1/devices/pools')).data;
        for (let tries = 0; tries < 40 && pools.left.every((c) => c.id !== two.id); tries += 1) pools = (await client.get('/api/v1/devices/pools')).data;
        expect(sharing(pools)).toBe(false);
        // The second computer joins the first one's network; nothing is seen until it is read again.
        twoApp.network = home('10.9.5.20');
        expect(sharing((await client.get('/api/v1/devices/pools')).data)).toBe(false);
        const checked = (await client.post('/api/v1/devices/pools/check', {})).data;
        expect(checked.checkedAt).toBeTruthy();
        expect(sharing(checked)).toBe(true);
        expect(checked.groups.find((group) => group.network === '10.9.5.0/24').computers.map((c) => c.address).sort()).toEqual(['10.9.5.10', '10.9.5.20']);
    });

    test('a hello that cannot be checked just now asks the app to try again instead of forgetting its link', async () => {
        const gateway = new DeviceGateway();
        const closed = [];
        const answer = async (verify) => {
            let handler = null;
            const ws = { once: (event, fn) => { handler = fn; }, close: (code) => closed.push(code), on: () => {}, send: () => {} };
            gateway.handleConnection(ws, verify);
            await handler(JSON.stringify({ type: 'hello', deviceId: 'x', key: 'y' }));
        };
        await answer(async () => { throw new Error('database did not answer'); });
        await answer(async () => null);
        expect(closed).toEqual([1011, 1008]);
    });

    test('online model names are made readable when the service gives none', () => {
        expect(cloudTitle('claude-sonnet-4-5-20250929')).toBe('Claude Sonnet 4.5');
        expect(cloudTitle('models/gemini-2.5-pro')).toBe('Gemini 2.5 Pro');
        expect(cloudTitle('gpt-oss:120b-cloud')).toBe('GPT OSS 120B');
        expect(cloudTitle('grok-4-latest')).toBe('Grok 4');
    });

    test('a computer is shown by a readable name until the person names it', async () => {
        await testStartup.loginAsUser('admin');
        const named = async (id) => (await listed()).find((device) => device.id === id).name;
        const first = await client.post('/api/v1/devices/link', { hostname: 'Davids-MacBook-Pro.local', hardware });
        expect(first.data.device.name).toBe("David's MacBook Pro");
        expect(await named(first.data.device.id)).toBe("David's MacBook Pro");

        // The desktop app sends the computer's own name, which takes the place of the network name.
        const again = await client.post('/api/v1/devices/link', { hostname: 'Davids-MacBook-Pro.local', name: 'Studio Mac', hardware });
        expect(again.data.device.id).toBe(first.data.device.id);
        expect(await named(first.data.device.id)).toBe('Studio Mac');

        // A name the person chose is never replaced.
        await client.put(`/api/v1/devices/${first.data.device.id}`, { name: 'My desk' });
        await client.post('/api/v1/devices/link', { hostname: 'Davids-MacBook-Pro.local', name: 'Studio Mac', hardware });
        expect(await named(first.data.device.id)).toBe('My desk');

        const plain = await client.post('/api/v1/devices/link', { hostname: 'zero-pc', hardware });
        expect(plain.data.device.name).toBe('zero-pc');
    });

    test('a computer that reconnects while still online is not announced as a change', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('reconnecting-computer');
        const service = getNotificationService();
        const sent = [];
        const real = service.sendToUser.bind(service);
        service.sendToUser = (userId, notification) => {
            if (notification.type === 'device:changed' && notification.data?.deviceId === id) sent.push(notification.data.online);
            return real(userId, notification);
        };
        const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
        try {
            await fakeApp(id, key);
            for (let attempt = 0; attempt < 50 && sent.length < 1; attempt++) await pause(10);
            expect(sent).toEqual([true]);
            await fakeApp(id, key);
            await pause(150);
            expect(sent).toEqual([true]);
        } finally {
            service.sendToUser = real;
        }
    });

    test('the models on a linked computer are listed for the chat picker', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('picker-computer');
        await fakeApp(id, key);
        const response = await client.client.get('/api/v1/models', through(id));
        expect(response.status).toBe(200);
        expect(response.data.ollamaAvailable).toBe(true);
        expect(response.data.models.map((model) => model.name)).toEqual(['tiny:1b']);
        expect(response.data.models[0].canChat).toBe(true);
        expect(response.data.device).toBe(id);
    });
    const until = async (check) => {
        for (let attempt = 0; attempt < 60; attempt += 1) {
            if (await check()) return true;
            await new Promise((resolve) => setTimeout(resolve, 50));
        }
        return false;
    };

    beforeAll(async () => {
        await Device.deleteMany({ kind: 'companion' });
    });

    beforeEach(() => {
        setOllamaClientFactory(simpleOllama());
    });

    afterAll(async () => {
        delete process.env.LOCAL_DEVICE;
        process.env.LOCAL_DEVICE = 'true';
        for (const ws of open) ws.close();
        await Device.deleteMany({ kind: 'companion' });
    });

    test('linking needs sign-in and a computer name', async () => {
        client.clearCookies();
        client.authenticatedAs = null;
        expect((await client.post('/api/v1/devices/link', { hostname: 'laptop' })).status).toBe(401);
        await testStartup.loginAsUser('user');
        expect((await client.post('/api/v1/devices/link', {})).status).toBe(400);
        expect((await client.post('/api/v1/devices/link', { hostname: 'laptop', hardware: { memoryMb: 'lots' } })).status).toBe(400);
    });

    test('a linked computer is listed for its owner as offline, and its key is never stored or shown', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link();
        expect(key).toMatch(/^[0-9a-f]{64}$/);

        const mine = (await listed()).find((device) => device.id === id);
        expect(mine).toMatchObject({ kind: 'companion', name: 'laptop', online: false, service: 'ai-base' });
        expect(mine.hardware.memoryMb).toBe(16384);
        expect(mine.ollama.available).toBe(false);
        expect(JSON.stringify(mine)).not.toContain(key);

        const stored = await Device.findById(id).select('+keyHash');
        expect(stored.keyHash).toHaveLength(64);
        expect(stored.keyHash).not.toBe(key);
        expect(String(stored.registeredBy)).toBe(String(testStartup.user.id || testStartup.user._id || stored.registeredBy));
    });

    test('another person cannot see, read, change or remove it', async () => {
        await testStartup.loginAsUser('user');
        const { id } = await link();

        await testStartup.loginAsUser('creator');
        expect((await listed()).some((device) => device.id === id)).toBe(false);
        expect((await client.get(`/api/v1/devices/${id}`)).status).toBe(404);
        expect((await client.delete(`/api/v1/devices/${id}`)).status).toBe(404);
        const status = await client.client.get('/api/v1/setup/status', through(id));
        expect(status.data.device?.id).not.toBe(id);

        await testStartup.loginAsUser('admin');
        expect((await client.put(`/api/v1/devices/${id}`, { name: 'Taken' })).status).toBe(404);
        expect(await Device.exists({ _id: id })).toBeTruthy();
    });

    test('the same computer name under two accounts makes two separate devices', async () => {
        await testStartup.loginAsUser('user');
        const first = await link('shared-name');
        await testStartup.loginAsUser('creator');
        const second = await link('shared-name');
        expect(second.id).not.toBe(first.id);
    });

    test('a wrong key, an unknown device and a silent connection are all turned away', async () => {
        await testStartup.loginAsUser('user');
        const { id } = await link();
        expect((await fakeApp(id, 'f'.repeat(64))).welcomed).toBe(false);
        expect((await fakeApp('0'.repeat(24), 'f'.repeat(64))).welcomed).toBe(false);
        expect((await fakeApp('not-an-id', 'x')).welcomed).toBe(false);
        expect((await listed()).find((device) => device.id === id).online).toBe(false);
    });

    test('with the right key the computer comes online and requests travel through it', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link();
        const app = await fakeApp(id, key);
        expect(app.welcomed).toBe(true);

        const mine = (await listed()).find((device) => device.id === id);
        expect(mine.online).toBe(true);
        expect(mine.ollama).toMatchObject({ available: true, version: '9.9.9', modelCount: 1 });

        const status = await client.client.get('/api/v1/setup/status', through(id));
        expect(status.status).toBe(200);
        expect(status.data.device).toMatchObject({ id, kind: 'companion' });
        expect(status.data.ollama).toMatchObject({ available: true, version: '9.9.9', modelCount: 1 });
        expect(status.data.steps).toEqual({ ollama: true, device: true, model: true });

        const running = await client.client.get('/api/v1/models/running', through(id));
        expect(running.status).toBe(200);
        expect(app.requests).toEqual(expect.arrayContaining(['GET /api/version', 'GET /api/tags', 'GET /api/ps']));

        const withoutHeader = await client.get('/api/v1/setup/status');
        expect(withoutHeader.data.device.kind).toBe('local');
    });

    test('one computer without Ollama is not a problem while another can run models', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('works');
        const app = await fakeApp(id, key);

        setOllamaClientFactory(simpleOllama({ up: false }));
        const status = await client.get('/api/v1/setup/status');
        expect(status.data.ollama).toMatchObject({ available: false, code: 'OLLAMA_DOWN' });
        expect(status.data.account).toEqual({ available: true });
        expect(status.data.ready).toBe(id);

        for (const ws of open) ws.close();
        await app.closed;
        expect(await until(async () => (await listed()).every((device) => device.kind === 'local' || !device.online))).toBe(true);
        const none = await client.get('/api/v1/setup/status');
        expect(none.data.ready).toBeNull();
        expect(none.data.account).toMatchObject({ available: false, code: 'NOTHING_READY' });
        expect(none.data.account.message).not.toMatch(/this device/i);
    });

    test('reading the hardware again asks the computer itself', async () => {
        await testStartup.loginAsUser('admin');
        const { id, key } = await link('workstation');
        await fakeApp(id, key);
        const response = await client.post(`/api/v1/devices/${id}/detect`, {});
        expect(response.status).toBe(200);
        expect(response.data.device.hardware.memoryMb).toBe(32768);
    });

    test('when the app disconnects the computer goes offline and requests to it fail plainly', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link();
        const app = await fakeApp(id, key);
        app.ws.close();
        await app.closed;
        expect(await until(async () => (await listed()).find((device) => device.id === id).online === false)).toBe(true);

        const status = await client.client.get('/api/v1/setup/status', through(id));
        expect(status.data.ollama).toMatchObject({ available: false, code: 'DEVICE_OFFLINE' });
        const models = await client.client.get('/api/v1/models/running', through(id));
        expect(models.status).toBe(503);
        expect(models.data.code).toBe('DEVICE_OFFLINE');
    });

    test('linking again replaces the key, so the old one stops working', async () => {
        await testStartup.loginAsUser('user');
        const first = await link();
        const app = await fakeApp(first.id, first.key);
        const second = await link();
        expect(second.id).toBe(first.id);
        expect(second.key).not.toBe(first.key);
        await app.closed;
        expect((await fakeApp(first.id, first.key)).welcomed).toBe(false);
        expect((await fakeApp(second.id, second.key)).welcomed).toBe(true);
    });

    test('signing out of the desktop app takes the computer off the account, but a dropped connection does not', async () => {
        await testStartup.loginAsUser('user');
        const kept = await link('just-offline');
        const dropped = await fakeApp(kept.id, kept.key);
        dropped.ws.close();
        await dropped.closed;
        expect(await until(async () => (await listed()).find((device) => device.id === kept.id)?.online === false)).toBe(true);

        const leaving = await link('signing-out');
        const app = await fakeApp(leaving.id, leaving.key);
        app.ws.send(JSON.stringify({ type: 'goodbye' }));
        app.ws.close();
        await app.closed;
        expect(await until(async () => !(await listed()).some((device) => device.id === leaving.id))).toBe(true);
        expect(await Device.exists({ _id: leaving.id })).toBeNull();
        expect((await fakeApp(leaving.id, leaving.key)).welcomed).toBe(false);
        expect(await Device.exists({ _id: kept.id })).toBeTruthy();
    });

    test('the same model can be downloading on two computers, each with its own list', async () => {
        await testStartup.loginAsUser('admin');
        const model = 'shared-model:1b';
        await Download.deleteMany({ model });
        const { id, key } = await link('second-computer');
        const app = await fakeApp(id, key, { holdPull: true });

        const standIn = String((await Device.findOne({ kind: 'local', service: 'ai-base' }))._id);
        const onServer = await client.client.post('/api/v1/models/pull', { name: model }, through(standIn));
        const onLinked = await client.client.post('/api/v1/models/pull', { name: model }, through(id));
        expect(onServer.status).toBe(202);
        expect(onLinked.status).toBe(202);
        expect(onLinked.data.download.id).not.toBe(onServer.data.download.id);
        expect(onServer.data.download.device).toBeNull();
        expect(onLinked.data.download.device).toBe(id);

        const again = await client.client.post('/api/v1/models/pull', { name: model }, through(id));
        expect(again.data.download.id).toBe(onLinked.data.download.id);

        const serverList = (await client.get('/api/v1/models/downloads')).data.downloads.filter((job) => job.model === model);
        const linkedList = (await client.client.get('/api/v1/models/downloads', through(id))).data.downloads.filter((job) => job.model === model);
        expect(serverList.map((job) => job.id)).toEqual([onServer.data.download.id]);
        expect(linkedList.map((job) => job.id)).toEqual([onLinked.data.download.id]);

        await testStartup.loginAsUser('creator');
        const others = (await client.client.get('/api/v1/models/downloads', through(id))).data.downloads;
        expect(others.some((job) => job.id === onLinked.data.download.id)).toBe(false);
        app.release();
        await downloads.settled(onServer.data.download.id).catch(() => {});
        await downloads.settled(onLinked.data.download.id).catch(() => {});
        await Download.deleteMany({ model });
    });

    test('a chat remembers its computer and uses it without being told again', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('chat-computer');
        const app = await fakeApp(id, key);

        const created = await client.post('/api/v1/chats', { device: id, model: 'tiny:1b' });
        expect(created.status).toBe(201);
        expect(created.data.chat.device).toBe(id);

        app.requests.length = 0;
        const opened = await client.get(`/api/v1/chats/${created.data.chat.id}`);
        expect(opened.data.chat.device).toBe(id);
        expect(app.requests.some((request) => request.startsWith('POST /api/show'))).toBe(true);

        const plain = await client.post('/api/v1/chats', { model: 'tiny:1b' });
        expect(plain.data.chat.device).toBeNull();
        app.requests.length = 0;
        await client.get(`/api/v1/chats/${plain.data.chat.id}`);
        expect(app.requests).toEqual([]);

        const cleared = await client.patch(`/api/v1/chats/${created.data.chat.id}`, { device: null });
        expect(cleared.data.chat.device).toBeNull();
        expect((await client.patch(`/api/v1/chats/${created.data.chat.id}`, { device: id })).data.chat.device).toBe(id);
    });

    test('a chat cannot be put on another person\'s computer', async () => {
        await testStartup.loginAsUser('user');
        const { id } = await link('not-yours');
        await testStartup.loginAsUser('creator');
        expect((await client.post('/api/v1/chats', { device: id })).status).toBe(404);
        const mine = await client.post('/api/v1/chats', {});
        expect((await client.patch(`/api/v1/chats/${mine.data.chat.id}`, { device: id })).status).toBe(404);
        expect((await client.patch(`/api/v1/chats/${mine.data.chat.id}`, { device: 'not-an-id' })).status).toBe(400);
    });

    test('a paused computer stays connected, is given no work, and works again when resumed', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('pause-me');
        const app = await fakeApp(id, key);

        app.requests.length = 0;
        const paused = await client.post(`/api/v1/devices/${id}/pause`, {});
        expect(paused.status).toBe(200);
        expect(paused.data.device).toMatchObject({ id, paused: true, online: true });
        expect(paused.data.device.ollama).toMatchObject({ available: false, paused: true });
        expect(app.requests).toContain('GET /api/ps');
        expect((await Device.findById(id)).paused).toBe(true);

        app.requests.length = 0;
        const mine = (await listed()).find((device) => device.id === id);
        expect(mine).toMatchObject({ paused: true, online: true });
        const refused = await client.client.get('/api/v1/models/running', through(id));
        expect(refused.status).toBe(503);
        expect(refused.data).toMatchObject({ code: 'DEVICE_PAUSED' });
        expect(refused.data.message).toMatch(/paused/i);
        const status = await client.client.get('/api/v1/setup/status', through(id));
        expect(status.data.ollama).toMatchObject({ available: false, code: 'DEVICE_PAUSED' });
        expect(app.requests).toEqual([]);

        const resumed = await client.post(`/api/v1/devices/${id}/resume`, {});
        expect(resumed.data.device).toMatchObject({ paused: false });
        expect(resumed.data.device.ollama).toMatchObject({ available: true });
        expect((await client.client.get('/api/v1/models/running', through(id))).status).toBe(200);

        await testStartup.loginAsUser('creator');
        expect((await client.post(`/api/v1/devices/${id}/pause`, {})).status).toBe(404);
        expect((await Device.findById(id)).paused).toBe(false);
    });

    // Stands in for a desktop app whose computer can chat and download, and whose connection can be cut part-way.
    const workingApp = (deviceId, key, { cutPull = false, cutChat = false } = {}) => new Promise((resolve) => {
        const ws = new WebSocket(`ws://localhost:${testStartup.port}/devices`);
        const app = { ws, requests: [], closed: new Promise((done) => ws.addEventListener('close', done)) };
        open.push(ws);
        const say = (message) => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify(message));
        const head = (id, status = 200, type = 'application/json') => say({ id, type: 'head', status, headers: { 'content-type': type } });
        const piece = (id, value) => say({ id, type: 'chunk', data: Buffer.from(typeof value === 'string' ? value : `${JSON.stringify(value)}\n`).toString('base64') });
        const whole = (id, body) => {
            head(id);
            piece(id, JSON.stringify(body));
            say({ id, type: 'end' });
        };
        ws.addEventListener('open', () => say({ type: 'hello', deviceId, key }));
        ws.addEventListener('close', () => resolve(app));
        ws.addEventListener('message', (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'welcome') return resolve(app);
            if (message.type === 'ask') return say({ id: message.id, type: 'answer', data: message.what === 'disk' ? { diskFreeMb: 900000, diskTotalMb: 1000000 } : { ...hardware, memoryMb: 32768 } });
            if (message.type !== 'request') return undefined;
            app.requests.push(`${message.method} ${message.path}`);
            if (message.path === '/api/version') return whole(message.id, { version: '9.9.9' });
            if (message.path === '/api/tags') return whole(message.id, TAGS);
            if (message.path === '/api/ps') return whole(message.id, { models: [] });
            if (message.path === '/api/show') return whole(message.id, { capabilities: ['completion'], details: { parameter_size: '1B' }, model_info: { 'tiny.context_length': 8192 } });
            if (message.path === '/api/pull') {
                head(message.id, 200, 'application/x-ndjson');
                piece(message.id, { status: 'pulling manifest' });
                piece(message.id, { status: 'downloading', digest: 'sha256:abc', total: 1000, completed: 400 });
                if (cutPull) return setTimeout(() => ws.close(), 60);
                piece(message.id, { status: 'downloading', digest: 'sha256:abc', total: 1000, completed: 1000 });
                piece(message.id, { status: 'success' });
                return say({ id: message.id, type: 'end' });
            }
            if (message.path === '/api/chat') {
                head(message.id, 200, 'application/x-ndjson');
                piece(message.id, { model: 'tiny:1b', message: { role: 'assistant', content: 'Half an ' }, done: false });
                if (cutChat) return setTimeout(() => ws.close(), 60);
                piece(message.id, { model: 'tiny:1b', message: { role: 'assistant', content: 'answer.' }, done: false });
                piece(message.id, { model: 'tiny:1b', message: { role: 'assistant', content: '' }, done: true, eval_count: 3, eval_duration: 1000000 });
                return say({ id: message.id, type: 'end' });
            }
            head(message.id, 404);
            piece(message.id, '{"error":"not found"}');
            return say({ id: message.id, type: 'end' });
        });
    });

    test('a download whose computer drops off fails plainly, then carries on when the computer returns', async () => {
        await Download.deleteMany({});
        await testStartup.loginAsUser('admin');
        const { id, key } = await link('drops-mid-download');
        const first = await workingApp(id, key, { cutPull: true });

        const started = await client.client.post('/api/v1/models/pull', { name: 'someone/unlisted:q4' }, through(id));
        expect(started.status).toBe(202);
        const jobId = started.data.download.id;
        await first.closed;
        expect(await until(async () => (await Download.findById(jobId)).status === 'failed')).toBe(true);
        const failed = await Download.findById(jobId);
        expect(failed.errorCode).toBe('DEVICE_OFFLINE');
        expect(failed.completedBytes).toBe(400);
        expect(new OllamaError(failed.errorCode).message).toMatch(/not connected/i);

        const second = await workingApp(id, key);
        expect(await until(async () => (await Download.findById(jobId)).status === 'completed')).toBe(true);
        expect(second.requests).toContain('POST /api/pull');
        const done = await Download.findById(jobId);
        expect(done.errorCode).toBeNull();
        expect(done.completedBytes).toBe(1000);
        expect(String(done.device)).toBe(id);
    });

    test('a download that failed for another reason is not started again when the computer returns', async () => {
        await Download.deleteMany({});
        await testStartup.loginAsUser('admin');
        const { id, key } = await link('no-auto-resume');
        const owner = (await Device.findById(id)).registeredBy;
        const old = await Download.create({ model: 'someone/old:q4', device: id, status: 'failed', errorCode: 'DEVICE_OFFLINE', requestedBy: owner, finishedAt: new Date(Date.now() - 2 * 60 * 60 * 1000) });
        const other = await Download.create({ model: 'someone/full:q4', device: id, status: 'failed', errorCode: 'DISK_FULL', requestedBy: owner, finishedAt: new Date() });
        const app = await workingApp(id, key);
        await new Promise((resolve) => setTimeout(resolve, 400));
        expect(app.requests).not.toContain('POST /api/pull');
        expect((await Download.findById(old.id)).status).toBe('failed');
        expect((await Download.findById(other.id)).status).toBe('failed');
    });

    test('a reply whose computer drops off keeps what was written and ends with a plain reason', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('drops-mid-reply');
        const app = await workingApp(id, key, { cutChat: true });
        const chat = (await client.post('/api/v1/chats', { device: id, model: 'tiny:1b' })).data.chat;

        const response = await client.post(`/api/v1/chats/${chat.id}/messages`, { content: 'say something long' }, { responseType: 'text', transformResponse: [(data) => data] });
        expect(response.status).toBe(200);
        const pieces = String(response.data).split('\n').filter(Boolean).map((line) => JSON.parse(line));
        expect(pieces.filter((piece) => piece.type === 'token').map((piece) => piece.text).join('')).toBe('Half an ');
        await app.closed;

        const saved = (await client.get(`/api/v1/chats/${chat.id}`)).data.chat;
        const last = saved.messages[saved.messages.length - 1];
        expect(last).toMatchObject({ role: 'assistant', content: 'Half an ' });
        expect(last.error).toMatchObject({ code: 'DEVICE_OFFLINE' });
        expect(last.error.message).toMatch(/not connected/i);
        expect(saved.replying).toBe(false);

        await workingApp(id, key);
        const again = await client.post(`/api/v1/chats/${chat.id}/messages`, { regenerate: true }, { responseType: 'text', transformResponse: [(data) => data] });
        expect(again.status).toBe(200);
        const after = (await client.get(`/api/v1/chats/${chat.id}`)).data.chat;
        expect(after.messages[after.messages.length - 1]).toMatchObject({ role: 'assistant', content: 'Half an answer.' });
        expect(after.messages[after.messages.length - 1].error).toBeNull();
    });

    test('removing a computer disconnects it and cancels its key', async () => {
        await testStartup.loginAsUser('user');
        const { id, key } = await link('to-remove');
        const app = await fakeApp(id, key);
        expect((await client.delete(`/api/v1/devices/${id}`)).status).toBe(200);
        await app.closed;
        expect(await Device.exists({ _id: id })).toBeNull();
        expect((await fakeApp(id, key)).welcomed).toBe(false);
    });

    test('the server\'s own computer cannot be removed', async () => {
        await testStartup.loginAsUser('admin');
        const local = (await listed()).find((device) => device.kind === 'local');
        expect((await client.delete(`/api/v1/devices/${local.id}`)).status).toBe(404);
    });

    test('an Apple-chip Mac shows its built-in graphics even when its app reported none', async () => {
        await testStartup.loginAsUser('user');
        const reported = { ...hardware, platform: 'darwin', gpus: [], gpuMemoryMb: 0 };
        const mac = await client.post('/api/v1/devices/link', { hostname: 'apple-mac', hardware: { ...reported, cpu: 'Apple M3 Pro', memoryMb: 18432 } });
        const intel = await client.post('/api/v1/devices/link', { hostname: 'intel-mac', hardware: { ...reported, cpu: 'Intel(R) Core(TM) i7-9750H', memoryMb: 16384 } });
        const devices = await listed();
        const shown = (id) => devices.find((device) => device.id === id).hardware;

        expect(shown(mac.data.device.id)).toMatchObject({ gpuName: 'Apple M3 Pro built-in graphics', gpuMemoryMb: 12288, hasGpu: true });
        expect(shown(intel.data.device.id)).toMatchObject({ gpuName: null, gpuMemoryMb: 0, hasGpu: false });
    });

    test('a browser on another address is allowed to send the chosen device with its requests', async () => {
        const origin = process.env.ALLOWED_ORIGINS.split(',')[0].trim();
        const preflight = await fetch(`http://localhost:${testStartup.port}/api/v1/devices`, {
            method: 'OPTIONS',
            headers: { Origin: origin, 'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'x-mollama-device' }
        });
        expect(preflight.status).toBe(200);
        expect(preflight.headers.get('access-control-allow-origin')).toBe(origin);
        expect(preflight.headers.get('access-control-allow-headers').toLowerCase()).toContain('x-mollama-device');
    });

    test('no running server counts as a computer, in development, staging or production, whatever its settings say', async () => {
        await testStartup.loginAsUser('admin');
        const kinds = async () => (await listed()).map((device) => device.kind);
        const before = { local: process.env.LOCAL_DEVICE, mode: process.env.NODE_ENV };
        try {
            expect(await kinds()).toContain('local');
            for (const mode of ['development', 'staging', 'production']) {
                process.env.NODE_ENV = mode;
                for (const setting of [undefined, '', 'false', 'true']) {
                    if (setting === undefined) delete process.env.LOCAL_DEVICE;
                    else process.env.LOCAL_DEVICE = setting;
                    expect(await kinds()).not.toContain('local');
                    const status = await client.get('/api/v1/setup/status');
                    expect(status.data.device?.kind).not.toBe('local');
                }
            }
        } finally {
            process.env.NODE_ENV = before.mode;
            process.env.LOCAL_DEVICE = before.local;
        }
    });

    test('with the server\'s own computer switched off, only linked computers count', async () => {
        await Device.deleteMany({ kind: 'companion' });
        process.env.LOCAL_DEVICE = 'false';
        try {
            await testStartup.loginAsUser('creator');
            expect(await listed()).toEqual([]);
            const none = await client.get('/api/v1/setup/status');
            expect(none.data.device).toBeNull();
            expect(none.data.ollama).toMatchObject({ available: false, code: 'NO_DEVICE' });
            expect(none.data.account).toMatchObject({ available: false, code: 'NO_DEVICE' });
            const refused = await client.get('/api/v1/models/running');
            expect(refused.status).toBe(503);
            expect(refused.data.code).toBe('NO_DEVICE');

            const { id, key } = await link('only-one');
            const offline = await client.get('/api/v1/setup/status');
            expect(offline.data.ollama.code).toBe('DEVICE_OFFLINE');

            await fakeApp(id, key);
            const devices = await listed();
            expect(devices.map((device) => device.kind)).toEqual(['companion']);
            const ready = await client.get('/api/v1/setup/status');
            expect(ready.data.device).toMatchObject({ id, kind: 'companion' });
            expect(ready.data.ollama.available).toBe(true);
            expect(ready.data.ready).toBe(id);
        } finally {
            process.env.LOCAL_DEVICE = 'true';
        }
    });
});

describe('Tool servers', () => {
    const any = { validateStatus: () => true };
    const api = '/api/v1/tools/servers';
    const TOOLS = [
        { name: 'find_page', title: 'Find a page', description: 'Finds a page by words in it', annotations: { readOnlyHint: true }, inputSchema: { type: 'object', properties: { words: { type: 'string' } } } },
        { name: 'write_page', description: 'Writes a page', inputSchema: { type: 'object' } }
    ];
    const hardware = { hostname: 'toolbox', platform: 'darwin', cpu: 'Test CPU', cpuThreads: 8, memoryMb: 16384, gpus: [], gpuMemoryMb: 0, diskFreeMb: 50000, diskTotalMb: 100000 };
    const calls = [];
    const open = [];
    let finished = [];

    // Stands in for the tool-server library: what an address does is read from its host name.
    const connector = async (address, options) => {
        const host = new URL(address).hostname;
        calls.push({ host, headers: options.requestInit?.headers || null, keeper: Boolean(options.authProvider) });
        if (host === 'down.example') throw new Error('connect failed');
        if (host === 'crowded.example') throw Object.assign(new Error('slow down'), { status: 429 });
        if (host === 'inside.example') throw new OllamaError('REMOTE_BLOCKED');
        if (host === 'keyed.example' && options.requestInit?.headers?.Authorization !== 'Bearer good-key') throw Object.assign(new Error('no'), { status: 401 });
        if (host === 'account.example' && !(await options.authProvider?.tokens())) {
            await options.authProvider?.redirectToAuthorization(new URL(`https://account.example/authorize?state=${await options.authProvider.state?.()}`));
            throw new ToolSignInError('sign in');
        }
        if (host === 'odd.example') return { listTools: async () => ({ tools: [{ name: 'fine', title: { en: 'Not text' }, description: 42, inputSchema: { type: 'object', big: 'x'.repeat(30000) } }, { title: 'No name' }, null, { name: 'listed', inputSchema: ['not', 'an', 'object'] }] }), close: async () => {} };
        return { listTools: async () => ({ tools: host === 'bare.example' ? [] : TOOLS }), close: async () => {} };
    };
    const finisher = async (address, keeper, params) => {
        finished.push(Object.fromEntries(params));
        if (params.get('code') !== 'good-code') throw new Error('bad code');
        await keeper.saveTokens({ access_token: 'secret-access-token', token_type: 'Bearer' });
    };
    const add = (body) => client.post(api, { kind: 'remote', ...body }, any);
    const signOut = () => {
        client.clearCookies();
        client.authenticatedAs = null;
    };

    beforeAll(async () => {
        setToolConnector(connector, finisher);
    });
    afterAll(async () => {
        setToolConnector();
        for (const ws of open) ws.close();
        await ToolServer.deleteMany({});
    });
    beforeEach(async () => {
        calls.length = 0;
        finished = [];
        await ToolServer.deleteMany({});
        signOut();
        await testStartup.loginAsUser('user');
    });

    test('sealing keeps a value unreadable and gives it back whole, and a damaged seal gives nothing', () => {
        const sealed = seal({ key: 'good-key' });
        expect(sealed).not.toContain('good-key');
        expect(unseal(sealed)).toEqual({ key: 'good-key' });
        expect(unseal(`${sealed.slice(0, -4)}AAAA`)).toBeNull();
        expect(seal({ key: 'good-key' })).not.toBe(sealed);
    });

    test('an app that lists its tools oddly is still added and checked, with only what can be kept', async () => {
        await testStartup.loginAsUser('user');
        const added = await add({ name: 'Odd app', address: 'https://odd.example/mcp' });
        expect(added.status).toBe(201);
        expect(added.data.server.health.state).toBe('healthy');
        expect(added.data.server.tools.map((tool) => tool.name)).toEqual(['fine', 'listed']);
        const checked = await client.post(`${api}/${added.data.server.id}/check`, {}, any);
        expect(checked.status).toBe(200);
        const stored = await ToolServer.findById(added.data.server.id);
        expect(stored.tools[0].toObject()).toMatchObject({ name: 'fine', title: null, description: '', input: { type: 'object' } });
        expect(stored.tools[1].input).toBeNull();
        await client.delete(`${api}/${added.data.server.id}`, any);
    });

    test('the list of online apps is served, and every entry is a named https address that can be added as it is', async () => {
        const response = await client.get('/api/v1/tools/catalogue', any);
        expect(response.status).toBe(200);
        expect(response.data.apps).toEqual(TOOL_CATALOGUE.map((entry) => ({ ...entry, keyNeeded: entry.signIn === 'key' })));
        expect(TOOL_CATALOGUE.length).toBeGreaterThanOrEqual(30);
        expect(new Set(TOOL_CATALOGUE.map((entry) => entry.name)).size).toBe(TOOL_CATALOGUE.length);
        expect(new Set(TOOL_CATALOGUE.map((entry) => entry.address)).size).toBe(TOOL_CATALOGUE.length);
        for (const entry of TOOL_CATALOGUE) {
            expect(toolSchemas.add.validate({ kind: 'remote', name: entry.name, address: entry.address }).error).toBeUndefined();
            expect(['account', 'key', 'none']).toContain(entry.signIn);
            if (entry.signIn === 'key') expect(Boolean(entry.key)).toBe(true);
            if (entry.signIn === 'account') expect(entry.key).toBeNull();
            if (entry.key) {
                expect(entry.key.page).toMatch(/^https:\/\//);
                expect(entry.key.steps.length).toBeGreaterThanOrEqual(3);
                expect(entry.key.cost.length).toBeGreaterThan(3);
            }
            expect(entry.does.length).toBeGreaterThan(10);
            expect(entry.group.length).toBeGreaterThan(3);
        }
        signOut();
        expect((await client.get('/api/v1/tools/catalogue', any)).status).toBe(401);
    });

    test('the suggested local apps are served, each with what starts it and what it needs first', async () => {
        const response = await client.get('/api/v1/tools/catalogue', any);
        expect(response.data.localApps).toEqual(TOOL_LOCAL_CATALOGUE);
        expect(TOOL_LOCAL_CATALOGUE.length).toBeGreaterThanOrEqual(12);
        expect(new Set(TOOL_LOCAL_CATALOGUE.map((entry) => entry.name)).size).toBe(TOOL_LOCAL_CATALOGUE.length);
        for (const entry of TOOL_LOCAL_CATALOGUE) {
            expect(entry.command).toMatch(/^[a-z]+$/);
            expect(entry.args).not.toMatch(/[;&|`$]/);
            expect(['maker', 'standard', 'community']).toContain(entry.by);
            expect(entry.needs.length).toBeGreaterThan(0);
            expect(entry.does.length).toBeGreaterThan(10);
            if (entry.by === 'community') expect(entry.caution).toMatch(/community/);
            if (entry.args.includes('<')) expect(entry.needs.join(' ')).toMatch(/Replace </);
        }
    });

    test('every health state has a label, and every state but healthy says what is needed', () => {
        expect(Object.keys(TOOL_HEALTH).sort()).toEqual([...TOOL_HEALTH_STATES].sort());
        for (const [state, entry] of Object.entries(TOOL_HEALTH)) {
            expect(entry.label.length).toBeGreaterThan(0);
            expect(entry.needs.length > 0).toBe(state !== 'healthy');
        }
    });

    test('a remote server that answers is healthy and lists its tools, with read-only ones marked', async () => {
        const response = await add({ name: 'Notes', address: 'https://open.example/mcp' });
        expect(response.status).toBe(201);
        const { server } = response.data;
        expect(server.health.state).toBe('healthy');
        expect(server.health.needs).toEqual([]);
        expect(server.tools).toEqual([
            { name: 'find_page', title: 'Find a page', description: 'Finds a page by words in it', readOnly: true },
            { name: 'write_page', title: null, description: 'Writes a page', readOnly: false }
        ]);
        expect(server.signIn).toBe('none');
        const listed = await client.get(api, any);
        expect(listed.data.servers.map((one) => one.name)).toEqual(['Notes']);
    });

    test('a key is sent to the server, stored sealed, and never returned', async () => {
        const response = await add({ name: 'Keyed', address: 'https://keyed.example/mcp', key: 'good-key' });
        expect(response.data.server.health.state).toBe('healthy');
        expect(response.data.server.signIn).toBe('key');
        expect(calls[0].headers).toEqual({ Authorization: 'Bearer good-key' });
        const stored = await ToolServer.findOne({ name: 'Keyed' }).select('+sealed');
        expect(stored.sealed).not.toContain('good-key');
        expect(JSON.stringify((await client.get(api, any)).data)).not.toContain('good-key');
        expect(JSON.stringify(response.data)).not.toContain('good-key');
    });

    test('a wrong key is reported as refused, and replacing it makes the server healthy', async () => {
        const response = await add({ name: 'Keyed', address: 'https://keyed.example/mcp', key: 'wrong' });
        expect(response.data.server.health.state).toBe('refused');
        const fixed = await client.patch(`${api}/${response.data.server.id}`, { key: 'good-key' }, any);
        expect(fixed.data.server.health.state).toBe('healthy');
    });

    test('unreachable, blocked and empty servers are saved with a state that says what is needed', async () => {
        const down = (await add({ name: 'Down', address: 'https://down.example/mcp' })).data.server;
        const inside = (await add({ name: 'Inside', address: 'https://inside.example/mcp' })).data.server;
        const bare = (await add({ name: 'Bare', address: 'https://bare.example/mcp' })).data.server;
        expect([down.health.state, inside.health.state, bare.health.state]).toEqual(['unreachable', 'blocked', 'empty']);
        for (const server of [down, inside, bare]) expect(server.health.needs.length).toBeGreaterThan(0);
        expect(down.tools).toEqual([]);
    });

    test('on a hosted site an app that offers free keys needs one, and elsewhere it can be added without', async () => {
        const exa = TOOL_CATALOGUE.find((entry) => entry.name === 'Exa');
        const open = TOOL_CATALOGUE.find((entry) => entry.name === 'DeepWiki');
        expect((await add({ name: exa.name, address: exa.address })).status).toBe(201);
        await ToolServer.deleteMany({});

        const usual = process.env.NODE_ENV;
        process.env.NODE_ENV = 'production';
        try {
            const listed = (await client.get('/api/v1/tools/catalogue', any)).data.apps;
            expect(listed.filter((entry) => entry.keyNeeded).map((entry) => entry.name).sort()).toEqual(['Context7', 'Exa', 'Firecrawl', 'GitHub', 'Hugging Face']);
            const refused = await add({ name: exa.name, address: exa.address });
            expect(refused.status).toBe(422);
            expect(refused.data.action).toMatch(/free key/);
            expect(await ToolServer.countDocuments({})).toBe(0);
            expect((await add({ name: open.name, address: open.address })).status).toBe(201);
            calls.length = 0;
            expect((await add({ name: exa.name, address: exa.address, key: 'exa-key' })).status).toBe(201);
            expect(calls[0].headers).toEqual({ 'x-api-key': 'exa-key' });
        } finally {
            process.env.NODE_ENV = usual;
        }
    });

    test('a server that says it is taking too many requests is busy, not unreachable, and the way out is a key', async () => {
        const crowded = (await add({ name: 'Crowded', address: 'https://crowded.example/mcp' })).data.server;
        expect(crowded.health.state).toBe('busy');
        expect(crowded.health.label).toBe('Busy right now');
        expect(crowded.health.needs.join(' ')).toMatch(/your own key/);
    });

    test('an address that is not https, or has no name, is refused before anything is saved', async () => {
        expect((await add({ name: 'Plain', address: 'http://open.example/mcp' })).status).toBe(400);
        expect((await add({ address: 'https://open.example/mcp' })).status).toBe(400);
        expect((await add({ name: 'Mixed', address: 'https://open.example/mcp', localId: 'abcdefgh' })).status).toBe(400);
        expect(await ToolServer.countDocuments({})).toBe(0);
    });

    test('a server that wants an account says sign-in is needed, and signing in makes it healthy', async () => {
        const added = (await add({ name: 'Account', address: 'https://account.example/mcp' })).data.server;
        expect(added.health.state).toBe('signin');
        expect(added.signIn).toBe('account');

        const started = await client.post(`${api}/${added.id}/sign-in`, {}, any);
        expect(started.status).toBe(200);
        const state = new URL(started.data.url).searchParams.get('state');
        expect(state.length).toBeGreaterThan(20);

        signOut();
        const back = await client.client.get(`/api/v1/tools/callback?code=good-code&state=${state}`, { maxRedirects: 0, validateStatus: () => true });
        expect(back.status).toBe(302);
        expect(back.headers.location).toBe(`${process.env.APP_URL}/tools?signin=done`);
        expect(finished).toEqual([{ code: 'good-code', state }]);

        await testStartup.loginAsUser('user');
        const after = (await client.get(api, any)).data.servers[0];
        expect(after.health.state).toBe('healthy');
        expect(after.tools.length).toBe(2);
        const stored = await ToolServer.findById(added.id).select('+sealed +signInState');
        expect(stored.sealed).not.toContain('secret-access-token');
        expect(unseal(stored.sealed).tokens.access_token).toBe('secret-access-token');
        expect(stored.signInState).toBeNull();
        expect(JSON.stringify(after)).not.toContain('secret-access-token');
    });

    test('a sign-in that comes back with the wrong state or a bad code changes nothing', async () => {
        const added = (await add({ name: 'Account', address: 'https://account.example/mcp' })).data.server;
        const started = await client.post(`${api}/${added.id}/sign-in`, {}, any);
        const state = new URL(started.data.url).searchParams.get('state');
        const options = { maxRedirects: 0, validateStatus: () => true };
        const wrong = await client.client.get('/api/v1/tools/callback?code=good-code&state=someone-elses', options);
        expect(wrong.headers.location).toBe(`${process.env.APP_URL}/tools?signin=failed`);
        const bad = await client.client.get(`/api/v1/tools/callback?code=bad&state=${state}`, options);
        expect(bad.headers.location).toBe(`${process.env.APP_URL}/tools?signin=failed`);
        const again = await client.client.get(`/api/v1/tools/callback?code=good-code&state=${state}`, options);
        expect(again.headers.location).toBe(`${process.env.APP_URL}/tools?signin=failed`);
        expect((await client.get(api, any)).data.servers[0].health.state).toBe('signin');
    });

    test('a server with no sign-in to offer says so plainly', async () => {
        const added = (await add({ name: 'Down', address: 'https://down.example/mcp' })).data.server;
        const started = await client.post(`${api}/${added.id}/sign-in`, {}, any);
        expect(started.status).toBe(422);
        expect(started.data.action).toMatch(/key/);
    });

    test('checking again picks up a change and tells the person over the notification channel', async () => {
        const added = (await add({ name: 'Notes', address: 'https://open.example/mcp' })).data.server;
        await ToolServer.updateOne({ _id: added.id }, { address: 'https://down.example/mcp' });
        const sent = [];
        const service = getNotificationService();
        const original = service.sendToUser;
        service.sendToUser = (userId, message) => { sent.push(message); };
        try {
            const checked = await client.post(`${api}/${added.id}/check`, {}, any);
            expect(checked.data.server.health.state).toBe('unreachable');
            expect(checked.data.server.tools.length).toBe(2);
            await client.post(`${api}/${added.id}/check`, {}, any);
        } finally {
            service.sendToUser = original;
        }
        expect(sent.filter((message) => message.type === 'tools:changed')).toEqual([{ type: 'tools:changed', data: { id: added.id, health: 'unreachable' } }]);
    });

    test('one person cannot see, check, change or remove the tool servers of another', async () => {
        const added = (await add({ name: 'Mine', address: 'https://open.example/mcp' })).data.server;
        signOut();
        await testStartup.loginAsUser('creator');
        expect((await client.get(api, any)).data.servers).toEqual([]);
        expect((await client.post(`${api}/${added.id}/check`, {}, any)).status).toBe(404);
        expect((await client.patch(`${api}/${added.id}`, { name: 'Theirs' }, any)).status).toBe(404);
        expect((await client.post(`${api}/${added.id}/sign-in`, {}, any)).status).toBe(404);
        expect((await client.delete(`${api}/${added.id}`, any)).status).toBe(404);
        expect((await ToolServer.findById(added.id)).name).toBe('Mine');
    });

    test('renaming and removing work, the limit is twenty, and a signed-out visitor is turned away', async () => {
        const added = (await add({ name: 'Notes', address: 'https://open.example/mcp' })).data.server;
        const owner = (await ToolServer.findById(added.id)).owner;
        expect((await client.patch(`${api}/${added.id}`, { name: 'Notebook' }, any)).data.server.name).toBe('Notebook');
        expect((await client.delete(`${api}/${added.id}`, any)).status).toBe(200);
        expect(await ToolServer.countDocuments({})).toBe(0);

        await ToolServer.insertMany(Array.from({ length: 20 }, (unused, index) => ({ owner, name: `s${index}`, kind: 'remote', address: 'https://open.example/mcp' })));
        expect((await add({ name: 'One more', address: 'https://open.example/mcp' })).status).toBe(422);

        signOut();
        expect((await client.get(api, any)).status).toBe(401);
    });

    describe('On a linked computer', () => {
        const fakeApp = (deviceId, key, answer) => new Promise((resolve) => {
            const ws = new WebSocket(`ws://localhost:${testStartup.port}/devices`);
            const app = { ws, asked: [] };
            open.push(ws);
            ws.addEventListener('open', () => ws.send(JSON.stringify({ type: 'hello', deviceId, key })));
            ws.addEventListener('message', (event) => {
                const message = JSON.parse(event.data);
                if (message.type === 'welcome') return resolve(app);
                if (message.type !== 'tools') return undefined;
                app.asked.push({ action: message.action, localId: message.localId });
                const data = answer(message);
                return data ? ws.send(JSON.stringify({ id: message.id, type: 'answer', data })) : undefined;
            });
        });
        const link = async () => {
            const response = await client.post('/api/v1/devices/link', { hostname: `toolbox-${Date.now()}`, hardware }, any);
            expect(response.status).toBe(201);
            return { id: response.data.device.id, key: response.data.key };
        };
        const addLocal = (device, localId = 'blender-1234') => client.post(api, { kind: 'local', name: 'Blender', device, localId }, any);

        test('a local server on a computer that is not connected is saved as offline', async () => {
            const { id } = await link();
            const response = await addLocal(id);
            expect(response.status).toBe(201);
            expect(response.data.server.health.state).toBe('offline');
            expect(response.data.server.kind).toBe('local');
            expect(response.data.server.address).toBeNull();
        });

        test('a connected computer is asked, and what its desktop app reports is recorded', async () => {
            const { id, key } = await link();
            let reply = { state: 'healthy', tools: [{ name: 'make_cube', description: 'Adds a cube', readOnly: false }] };
            const app = await fakeApp(id, key, () => reply);
            const added = (await addLocal(id)).data.server;
            expect(app.asked).toEqual([{ action: 'check', localId: 'blender-1234' }]);
            expect(added.health.state).toBe('healthy');
            expect(added.tools.map((tool) => tool.name)).toEqual(['make_cube']);

            reply = { state: 'stopped', detail: 'The program uvx was not found.' };
            const checked = (await client.post(`${api}/${added.id}/check`, {}, any)).data.server;
            expect(checked.health.state).toBe('stopped');
            expect(checked.health.detail).toBe('The program uvx was not found.');
            expect(checked.tools.map((tool) => tool.name)).toEqual(['make_cube']);
            expect((await addLocal(id)).status).toBe(409);
        });

        test('a desktop app too old to answer is reported as needing an update', async () => {
            const { id, key } = await link();
            await fakeApp(id, key, () => null);
            const usual = deviceGateway.askMs;
            deviceGateway.askMs = 200;
            try {
                expect((await addLocal(id)).data.server.health.state).toBe('outdated');
            } finally {
                deviceGateway.askMs = usual;
            }
        });

        test('a computer that belongs to someone else cannot be used', async () => {
            const { id } = await link();
            signOut();
            await testStartup.loginAsUser('creator');
            expect((await addLocal(id)).status).toBe(404);
        });
    });
});

describe('Online model providers', () => {
    const any = { validateStatus: () => true };
    const api = '/api/v1/providers';
    const asked = [];
    const through = (id) => ({ headers: { 'X-Mollama-Device': id }, validateStatus: () => true });
    const signOut = () => {
        client.clearCookies();
        client.authenticatedAs = null;
    };
    const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
    const stream = (events) => new Response(new ReadableStream({
        start(controller) {
            for (const event of events) controller.enqueue(new TextEncoder().encode(`data: ${typeof event === 'string' ? event : JSON.stringify(event)}\n\n`));
            controller.close();
        }
    }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } });

    // Stands in for every online service: what a key does is read from the key itself.
    const service = async (address, init = {}) => {
        const url = new URL(String(address));
        const key = new Headers(init.headers || {}).get('authorization')?.replace('Bearer ', '') || '';
        const body = init.body ? JSON.parse(init.body) : null;
        asked.push({ host: url.host, path: url.pathname, key, body });
        if (key === 'bad-key-0000') return json({ error: { message: 'invalid key' } }, 401);
        if (key === 'odd-key-0000') return json({ error: { code: 400, message: 'Please pass a valid API key' } }, 400);
        if (key === 'poor-key-0000') return json({ error: { message: 'insufficient balance' } }, 402);
        if (key === 'slow-key-0000') return json({ error: { message: 'slow down' } }, 429);
        if (url.host === 'api.anthropic.com') {
            asked.at(-1).version = new Headers(init.headers || {}).get('anthropic-version');
            if (key !== 'good-key-0000') return json({ type: 'error', error: { type: 'authentication_error', message: 'invalid x-api-key' } }, 401);
            if (url.pathname === '/v1/models') return json({ data: [{ type: 'model', id: 'claude-large', display_name: 'Claude Large', max_input_tokens: 1000000, max_tokens: 128000, capabilities: { image_input: { supported: true } } }], has_more: false, first_id: 'claude-large', last_id: 'claude-large' });
            if (url.pathname === '/v1/messages') {
                // "Work it out" makes Claude stream a calculator call in pieces; "cut" stops it at the length limit mid-call.
                const last = body.messages.at(-1);
                const wanted = typeof last?.content === 'string' && last.content.includes('Work it out');
                if (body.tools && wanted) {
                    const cut = last.content.includes('cut');
                    const calling = [
                        ['message_start', { type: 'message_start', message: { id: 'msg_2', type: 'message', role: 'assistant', model: body.model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 30, output_tokens: 1 } } }],
                        ['content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }],
                        ['content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Let me work it out. ' } }],
                        ['content_block_stop', { type: 'content_block_stop', index: 0 }],
                        ['content_block_start', { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 'toolu_a', name: 'calculator', input: {} } }],
                        ['content_block_delta', { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '{"expres' } }],
                        ['content_block_delta', { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: 'sion":"6 * 7"}' } }],
                        ['content_block_stop', { type: 'content_block_stop', index: 1 }],
                        ['message_delta', { type: 'message_delta', delta: { stop_reason: cut ? 'max_tokens' : 'tool_use', stop_sequence: null }, usage: { output_tokens: 20 } }],
                        ['message_stop', { type: 'message_stop' }]
                    ];
                    return new Response(calling.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join(''), { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
                }
                const events = [
                    ['message_start', { type: 'message_start', message: { id: 'msg_1', type: 'message', role: 'assistant', model: body.model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 30, output_tokens: 1 } } }],
                    ['content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '', signature: '' } }],
                    ['content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Considering. ' } }],
                    ['content_block_stop', { type: 'content_block_stop', index: 0 }],
                    ['content_block_start', { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '' } }],
                    ['content_block_delta', { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Hello from ' } }],
                    ['content_block_delta', { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Claude.' } }],
                    ['content_block_stop', { type: 'content_block_stop', index: 1 }],
                    ['message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 12 } }],
                    ['message_stop', { type: 'message_stop' }]
                ];
                return new Response(events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join(''), { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
            }
        }
        if (url.pathname.endsWith('/api/tags')) return json({ models: [{ name: 'kimi-k3', capabilities: ['completion', 'vision'] }, { name: 'embed-one', capabilities: ['embedding'] }] });
        if (url.pathname.endsWith('/models')) {
            return json({ data: key === 'none-key-0000' ? [] : [
                { id: 'big-one', context_length: 200000, supports_image_in: true },
                { id: 'models/small-one' },
                { id: 'text-embedding-3' },
                { id: 'whisper-9' }
            ] });
        }
        if (url.pathname.endsWith('/chat/completions')) {
            if (!body.stream) return json({ choices: [{ message: { content: 'A short summary.' } }], usage: { prompt_tokens: 9, completion_tokens: 4 } });
            return stream([
                { choices: [{ delta: { reasoning_content: 'Let me think. ' } }] },
                { choices: [{ delta: { content: 'Hello ' } }] },
                { choices: [{ delta: { content: [{ type: 'text', text: 'from online.' }] } }] },
                { choices: [], usage: { prompt_tokens: 21, completion_tokens: 7 } },
                '[DONE]'
            ]);
        }
        return json({ error: { message: 'no such path' } }, 404);
    };
    const add = (kind, key = 'good-key-0000') => client.post(api, { kind, key }, any);

    beforeAll(() => setCloudFetch(service));
    afterAll(async () => {
        setCloudFetch();
        await Provider.deleteMany({});
    });
    beforeEach(async () => {
        asked.length = 0;
        await Provider.deleteMany({});
        signOut();
        await testStartup.loginAsUser('user');
    });

    test('every suggested service has a secure address, a key page and steps, and every state says what is needed', () => {
        expect(CLOUD_PROVIDERS.length).toBeGreaterThanOrEqual(10);
        expect(CLOUD_PROVIDERS[0].id).toBe('ollama');
        expect(new Set(CLOUD_PROVIDERS.map((entry) => entry.id)).size).toBe(CLOUD_PROVIDERS.length);
        for (const entry of CLOUD_PROVIDERS) {
            expect(entry.base).toMatch(/^https:\/\/[^/]+/);
            expect(entry.base.endsWith('/')).toBe(false);
            expect(entry.key.page).toMatch(/^https:\/\//);
            expect(entry.key.steps.length).toBe(3);
            expect(providerSchemas.add.validate({ kind: entry.id, key: 'good-key-0000' }).error).toBeUndefined();
        }
        expect(Object.keys(PROVIDER_HEALTH).sort()).toEqual([...PROVIDER_HEALTH_STATES].sort());
        for (const [state, entry] of Object.entries(PROVIDER_HEALTH)) expect(entry.needs.length > 0).toBe(state !== 'healthy');
        for (const code of ['KEY_REFUSED', 'OUT_OF_CREDIT', 'PROVIDER_BUSY', 'PROVIDER_DOWN']) expect(new OllamaError(code).code).toBe(code);
    });

    test('adding a service with a working key lists its chat models, and the key is sealed and never returned', async () => {
        const response = await add('openai');
        expect(response.status).toBe(201);
        const { provider } = response.data;
        expect(provider.health.state).toBe('healthy');
        expect(provider.models).toEqual(['big-one', 'small-one']);
        expect(asked[0]).toMatchObject({ host: 'api.openai.com', path: '/v1/models', key: 'good-key-0000' });
        const stored = await Provider.findById(provider.id).select('+sealed');
        expect(stored.sealed).not.toContain('good-key-0000');
        expect(unseal(stored.sealed)).toEqual({ key: 'good-key-0000' });
        const listed = await client.get(api, any);
        expect(JSON.stringify(listed.data)).not.toContain('good-key-0000');
        expect(JSON.stringify(response.data)).not.toContain('good-key-0000');
        expect(listed.data.providers.map((one) => one.name)).toEqual(['OpenAI']);
        expect(listed.data.suggested.map((one) => one.id)).toEqual(CLOUD_PROVIDERS.map((one) => one.id));
        expect(listed.data.suggested[0].base).toBeUndefined();
    });

    test('Ollama online is asked the way Ollama is asked, and models that only make embeddings are left out', async () => {
        const { provider } = (await add('ollama')).data;
        expect(provider.health.state).toBe('healthy');
        expect(provider.models).toEqual(['kimi-k3']);
        expect(asked.map((one) => `${one.host}${one.path}`)).toEqual(['ollama.com/api/me', 'ollama.com/api/tags']);
        expect(asked.every((one) => one.key === 'good-key-0000')).toBe(true);
    });

    test('a service that lists its models to anyone is still asked to prove the key, and one that answers a bad key oddly is understood', async () => {
        expect((await add('ollama', 'bad-key-0000')).data.provider.health.state).toBe('refused');
        expect((await add('openrouter', 'bad-key-0000')).data.provider.health.state).toBe('refused');
        expect(asked.some((one) => one.host === 'openrouter.ai' && one.path === '/api/v1/key')).toBe(true);
        expect((await add('google', 'odd-key-0000')).data.provider.health.state).toBe('refused');
    });

    test('the models Ollama runs online are read without a key, and a library entry is matched by its exact size or its bare name', async () => {
        setCloudFetch(service);
        const names = await ollamaOnlineNames();
        expect([...names]).toEqual(['kimi-k3', 'embed-one']);
        expect(asked.at(-1)).toMatchObject({ host: 'ollama.com', path: '/api/tags', key: '' });
        asked.length = 0;
        await ollamaOnlineNames();
        expect(asked.length).toBe(0);

        const listed = new Set(['gpt-oss:120b', 'kimi-k3', 'deepseek-v4-pro:0813']);
        expect(runsOnline({ tag: 'gpt-oss:120b' }, listed)).toBe(true);
        expect(runsOnline({ tag: 'gpt-oss:20b' }, listed)).toBe(false);
        expect(runsOnline({ tag: 'kimi-k3', cloudOnly: true }, listed)).toBe(true);
        expect(runsOnline({ tag: 'deepseek-v4-pro', cloudOnly: true }, listed)).toBe(true);
        expect(runsOnline({ tag: 'deepseek-v4-pro' }, listed)).toBe(false);
        expect(runsOnline({ tag: 'llama3.2:3b' }, listed)).toBe(false);
    });

    test('a refused key, an empty account, no credit and a busy service are told apart, and a new key puts it right', async () => {
        const refused = (await add('openai', 'bad-key-0000')).data.provider;
        const credit = (await add('deepseek', 'poor-key-0000')).data.provider;
        const busy = (await add('groq', 'slow-key-0000')).data.provider;
        const empty = (await add('mistral', 'none-key-0000')).data.provider;
        expect([refused.health.state, credit.health.state, busy.health.state, empty.health.state]).toEqual(['refused', 'credit', 'busy', 'empty']);
        for (const one of [refused, credit, busy, empty]) expect(one.health.needs.length).toBeGreaterThan(0);
        const fixed = await client.patch(`${api}/${refused.id}`, { key: 'good-key-0000' }, any);
        expect(fixed.data.provider.health.state).toBe('healthy');
        expect((await client.post(`${api}/${credit.id}/check`, {}, any)).data.provider.health.state).toBe('credit');
    });

    test('an unknown service, a second copy of the same one and a key that is too short are refused', async () => {
        expect((await add('nowhere')).status).toBe(404);
        expect((await add('openai')).status).toBe(201);
        expect((await add('openai')).status).toBe(409);
        expect((await client.post(api, { kind: 'groq', key: 'x' }, any)).status).toBe(400);
        expect((await client.post(api, { kind: 'groq' }, any)).status).toBe(400);
    });

    test('a connected service appears among the computers only when asked for, with its models ready to chat', async () => {
        const { provider } = (await add('moonshot')).data;
        const plain = (await client.get('/api/v1/devices', any)).data.devices;
        expect(plain.some((device) => device.kind === 'cloud')).toBe(false);
        const withOnline = (await client.get('/api/v1/devices?with=online', any)).data.devices;
        const entry = withOnline.find((device) => device.id === provider.id);
        expect(entry).toMatchObject({ kind: 'cloud', name: 'Moonshot (online)', online: true, paused: false, ollama: { available: true, modelCount: 2 } });

        const models = (await client.client.get('/api/v1/models', through(provider.id))).data;
        expect(models.online).toBe(true);
        expect(models.device).toBe(provider.id);
        expect(models.models.map((model) => model.displayName)).toEqual(['Big One', 'Small One']);
        expect(models.models.map((model) => model.name)).toEqual(['big-one', 'small-one']);
        expect(models.models[0]).toMatchObject({ ready: true, canChat: true, contextLength: 200000, fit: { verdict: 'cloud', label: 'Online', fits: true } });
        expect(models.models[0].capabilities).toContain('vision');

        // The app's watcher must not move a person off an online service: it counts as ready.
        const status = (await client.client.get('/api/v1/setup/status', through(provider.id))).data;
        expect(status.ready).toBe(provider.id);
        expect(status.account.available).toBe(true);
    });

    test('a chat on an online model streams the reply, shows its thinking, counts its tokens and sends only the conversation', async () => {
        const { provider } = (await add('moonshot')).data;
        const made = await client.post('/api/v1/chats', { model: 'big-one', device: provider.id }, any);
        expect(made.status).toBe(201);
        expect(made.data.chat.device).toBe(provider.id);
        asked.length = 0;
        const sent = await client.client.post(`/api/v1/chats/${made.data.chat.id}/messages`, { content: 'Say hello' }, { responseType: 'text', validateStatus: () => true });
        expect(sent.status).toBe(200);
        const events = String(sent.data).trim().split('\n').map((line) => JSON.parse(line));
        expect(events.filter((event) => event.type === 'thinking').map((event) => event.text).join('')).toBe('Let me think. ');
        expect(events.filter((event) => event.type === 'token').map((event) => event.text).join('')).toBe('Hello from online.');
        const done = events.find((event) => event.type === 'done');
        expect(done.reply.content).toBe('Hello from online.');
        expect(done.reply.stats.evalCount).toBe(7);
        expect(done.reply.stats.promptEvalCount).toBe(21);

        const call = asked.find((one) => one.path.endsWith('/chat/completions'));
        expect(call).toMatchObject({ host: 'api.moonshot.ai', key: 'good-key-0000' });
        expect(call.body.model).toBe('big-one');
        expect(call.body.stream).toBe(true);
        expect(call.body.messages.at(-1)).toEqual({ role: 'user', content: 'Say hello' });
        // Besides the conversation, the functions every chat is offered: the search of the skill library and the built-in tools.
        expect(Object.keys(call.body).sort()).toEqual(['messages', 'model', 'stream', 'stream_options', 'tools']);
        expect(call.body.tools.map((tool) => tool.function.name)).toEqual(['open_component', 'open_skill_file', 'open_skill', 'find_skill', 'ask_user', 'save_note', 'list_my_files', 'read_my_file', 'save_to_files', 'date_time', 'calculator', 'read_web_page']);
        expect(call.body.options).toBeUndefined();
        expect(call.body.keep_alive).toBeUndefined();

        // An online model is measured against what its own service says it reads, not the limit kept for a computer's memory.
        expect(done.context.capacityTokens).toBeGreaterThan(32768);
        expect(done.context.percent).toBeLessThan(5);
    });

    test('Claude is reached through its own library: a dated request, its own request shape, and its thinking shown', async () => {
        const { provider } = (await add('anthropic')).data;
        expect(provider.health.state).toBe('healthy');
        expect(provider.models).toEqual(['claude-large']);
        expect(asked[0]).toMatchObject({ host: 'api.anthropic.com', path: '/v1/models', key: 'good-key-0000' });
        expect(asked[0].version).toMatch(/^\d{4}-\d{2}-\d{2}$/);
        expect((await Provider.findById(provider.id)).models[0]).toMatchObject({ contextLength: 1000000, maxOutput: 128000, vision: true });

        const made = await client.post('/api/v1/chats', { model: 'claude-large', device: provider.id }, any);
        asked.length = 0;
        const sent = await client.client.post(`/api/v1/chats/${made.data.chat.id}/messages`, { content: 'Say hello' }, { responseType: 'text', validateStatus: () => true });
        const events = String(sent.data).trim().split('\n').map((line) => JSON.parse(line));
        expect(events.filter((event) => event.type === 'thinking').map((event) => event.text).join('')).toBe('Considering. ');
        expect(events.filter((event) => event.type === 'token').map((event) => event.text).join('')).toBe('Hello from Claude.');
        const done = events.find((event) => event.type === 'done');
        expect(done.reply.stats).toMatchObject({ evalCount: 12, promptEvalCount: 30 });

        const call = asked.find((one) => one.path === '/v1/messages');
        expect(call.body.model).toBe('claude-large');
        expect(call.body.max_tokens).toBe(64000);
        expect(call.body.stream).toBe(true);
        expect(typeof call.body.system).toBe('string');
        expect(call.body.messages.every((message) => message.role !== 'system')).toBe(true);
        expect(call.body.messages.at(-1)).toEqual({ role: 'user', content: 'Say hello' });
        expect(call.body.temperature).toBeUndefined();

        expect((await add('anthropic', 'bad-key-0000')).status).toBe(409);
        await Provider.deleteMany({});
        expect((await add('anthropic', 'bad-key-0000')).data.provider.health.state).toBe('refused');
    });

    test('a call Claude is writing is shown as it streams, and a call cut off by the length limit is never run', async () => {
        const { provider } = (await add('anthropic')).data;
        const made = await client.post('/api/v1/chats', { model: 'claude-large', device: provider.id }, any);
        const say = async (content) => {
            const sent = await client.client.post(`/api/v1/chats/${made.data.chat.id}/messages`, { content }, { responseType: 'text', validateStatus: () => true });
            return String(sent.data).trim().split('\n').map((line) => JSON.parse(line));
        };
        asked.length = 0;
        const events = await say('Work it out: 6 * 7');
        // Every tool is sent with eager input streaming, and the call's input reaches the page piece by piece.
        const first = asked.find((one) => one.path === '/v1/messages');
        expect(first.body.tools.length).toBeGreaterThan(0);
        expect(first.body.tools.every((tool) => tool.eager_input_streaming === true)).toBe(true);
        expect(events.filter((event) => event.type === 'draft').map((event) => event.text)).toEqual(['{"expres', 'sion":"6 * 7"}']);
        expect(events.find((event) => event.type === 'tool')).toMatchObject({ tool: 'Calculator' });
        const second = asked.filter((one) => one.path === '/v1/messages')[1];
        expect(JSON.stringify(second.body.messages.at(-1))).toContain('42');

        // Cut off at the length limit, the half-written call is not run; the model is asked to go on instead.
        asked.length = 0;
        const cut = await say('Work it out, but this one gets cut');
        expect(cut.some((event) => event.type === 'tool')).toBe(false);
        // First it is given more room and asked again, which is cut the same way; then it is asked to go on.
        const calls = asked.filter((one) => one.path === '/v1/messages');
        expect(calls).toHaveLength(3);
        expect(JSON.stringify(calls[2].body.messages.at(-1))).toContain('Go on from exactly where you stopped');
    });

    test('a key that stops working mid-use ends the reply with a plain reason', async () => {
        const { provider } = (await add('openai')).data;
        const made = await client.post('/api/v1/chats', { model: 'big-one', device: provider.id }, any);
        await Provider.updateOne({ _id: provider.id }, { sealed: seal({ key: 'bad-key-0000' }) });
        const sent = await client.client.post(`/api/v1/chats/${made.data.chat.id}/messages`, { content: 'Hello?' }, { responseType: 'text', validateStatus: () => true });
        const events = String(sent.data).trim().split('\n').map((line) => JSON.parse(line));
        const failed = events.find((event) => event.type === 'error');
        expect(failed.code).toBe('KEY_REFUSED');
        expect(failed.message).toMatch(/did not accept your key/);
    });

    test('one person cannot see, use, check, change or remove a service another person connected', async () => {
        const { provider } = (await add('openai')).data;
        signOut();
        await testStartup.loginAsUser('creator');
        expect((await client.get(api, any)).data.providers).toEqual([]);
        expect((await client.get('/api/v1/devices?with=online', any)).data.devices.some((device) => device.id === provider.id)).toBe(false);
        expect((await client.post(`${api}/${provider.id}/check`, {}, any)).status).toBe(404);
        expect((await client.patch(`${api}/${provider.id}`, { key: 'good-key-0000' }, any)).status).toBe(404);
        expect((await client.delete(`${api}/${provider.id}`, any)).status).toBe(404);
        const models = (await client.client.get('/api/v1/models', through(provider.id))).data;
        expect(models.online).toBeUndefined();
        expect((await Provider.findById(provider.id)).name).toBe('OpenAI');
    });

    test('removing a service works, and a signed-out visitor is turned away', async () => {
        const { provider } = (await add('groq')).data;
        expect((await client.delete(`${api}/${provider.id}`, any)).status).toBe(200);
        expect(await Provider.countDocuments({})).toBe(0);
        signOut();
        expect((await client.get(api, any)).status).toBe(401);
    });
});

describe('Tools in chats', () => {
    const any = { validateStatus: () => true };
    const post = (path, body, config) => client.post(`/api/v1${path}`, body, { validateStatus: () => true, ...config });
    const called = [];
    const signOut = () => {
        client.clearCookies();
        client.authenticatedAs = null;
    };
    const SERVER_TOOLS = [
        { name: 'find_page', title: 'Find a page', description: 'Finds a page', annotations: { readOnlyHint: true }, inputSchema: { type: 'object', properties: { words: { type: 'string' } } } },
        { name: 'write_page', description: 'Writes a page', inputSchema: { type: 'object', properties: { text: { type: 'string' } } } }
    ];
    const connector = async () => ({
        listTools: async () => ({ tools: SERVER_TOOLS }),
        callTool: async (params) => {
            called.push(params);
            if (params.arguments?.words === 'broken') return { isError: true, content: [{ type: 'text', text: 'No page has those words.' }] };
            if (params.arguments?.words === 'huge') return { content: [{ type: 'text', text: 'x'.repeat(50000) }] };
            if (params.arguments?.words === 'slow') throw new Error('Request timed out');
            return { content: [{ type: 'text', text: params.name === 'find_page' ? 'Found: the launch plan.' : 'Saved.' }] };
        },
        close: async () => {}
    });
    const newChat = async (body = {}) => (await post('/chats', { model: 'qwen2.5-coder:7b', ...body })).data.chat;
    const send = async (chatId, content = 'do it') => {
        const response = await post(`/chats/${chatId}/messages`, { content }, { responseType: 'text', transformResponse: [(data) => data] });
        return String(response.data).trim().split('\n').map((line) => JSON.parse(line));
    };
    const until = async (check) => {
        for (let attempt = 0; attempt < 100; attempt += 1) {
            if (await check()) return true;
            await new Promise((resolve) => setTimeout(resolve, 30));
        }
        return false;
    };
    const addServer = async () => (await post('/tools/servers', { kind: 'remote', name: 'Notes', address: 'https://notes.example/mcp' })).data.server;
    const toolTurns = (request) => request.messages.filter((message) => message.role === 'tool');

    beforeAll(() => {
        setOllamaClientFactory(fakeOllama);
        setToolConnector(connector);
    });
    afterAll(async () => {
        setToolConnector();
        await ToolServer.deleteMany({});
    });
    beforeEach(async () => {
        resetState();
        called.length = 0;
        await ToolServer.deleteMany({});
        signOut();
        await testStartup.loginAsUser('user');
    });

    test('the built-in tools work on their own: sums are exact, nothing but sums is run, and a bad time zone is said plainly', async () => {
        const tool = (id) => BUILTIN_TOOLS.find((one) => one.id === id);
        expect(await tool('calculator').run({ expression: '(12.5 * 4) / 5 ^ 2' })).toBe('2');
        expect(await tool('calculator').run({ expression: '1,000 + 1' })).toBe('1001');
        expect(await tool('calculator').run({ expression: 'process.exit()' })).toMatch(/not an arithmetic expression/);
        expect(await tool('calculator').run({ expression: '1/0' })).toMatch(/not a finite number/);
        expect(await tool('date_time').run({ timezone: 'Europe/London' })).toMatch(/\d{4}/);
        expect(await tool('date_time').run({ timezone: 'Nowhere/Place' })).toMatch(/not one I know/);
        expect(await tool('read_web_page').run({ url: 'http://plain.example' })).toMatch(/https/);
    });

    test('what a chat can switch on is listed, and the choice is kept with the chat', async () => {
        const server = await addServer();
        const listed = (await client.get('/api/v1/skills', any)).data.tools;
        expect(listed.filter((tool) => tool.kind === 'builtin').map((tool) => tool.id)).toEqual(['date_time', 'calculator', 'read_web_page']);
        expect(listed.find((tool) => tool.id === server.id)).toMatchObject({ name: 'Notes', kind: 'remote', ready: true, count: 2 });

        const chat = await newChat({ tools: ['calculator', server.id] });
        expect(chat.tools).toEqual(['calculator', server.id]);
        const changed = await client.patch(`/api/v1/chats/${chat.id}`, { tools: [] }, any);
        expect(changed.data.chat.tools).toEqual([]);
        expect((await post('/chats', { tools: ['Not Valid!'] })).status).toBe(400);
    });

    test('the built-in tools are always offered, run when called, and their answers reach the model', async () => {
        // Nothing is switched on for this chat: the built-in tools need no switch.
        const chat = await newChat();
        state.toolsSeen = [];
        state.toolPlan = [[['calculator', { expression: '6 * 7' }]]];
        const events = await send(chat.id, 'what is six times seven');
        expect(state.toolsSeen[0].map((tool) => tool.function.name)).toEqual(['open_component', 'open_skill_file', 'open_skill', 'find_skill', 'ask_user', 'save_note', 'list_my_files', 'read_my_file', 'save_to_files', 'date_time', 'calculator', 'read_web_page']);
        expect(JSON.stringify(state.chatCalls[0].messages)).not.toContain('You have tools for this chat');
        expect(events.find((event) => event.type === 'tool')).toMatchObject({ tool: 'Calculator', input: '{"expression":"6 * 7"}' });
        expect(toolTurns(state.chatCalls[1])).toEqual([{ role: 'tool', tool_name: 'calculator', content: '42' }]);
        const done = events.find((event) => event.type === 'done');
        expect(done.reply.steps).toEqual([expect.objectContaining({ tool: 'Calculator', failed: null })]);
    });

    test('a chat adds no tools of its own unless a connected one is switched on, and a model that cannot use tools is offered none at all', async () => {
        const plain = await newChat();
        await send(plain.id);
        expect(state.chatCalls[0].tools).toBeUndefined();

        resetState();
        state.noTools = true;
        state.toolsSeen = [];
        const chat = await newChat({ tools: ['calculator'] });
        await send(chat.id);
        expect(state.toolsSeen).toEqual([null]);
    });

    test('past the hourly limit on tool calls a call is not run and the model is told why, and an hour later calls run again', async () => {
        // The real limit is 120 calls a person an hour; 2 shows the same rule. Only the clock is faked, so the server's
        // own timers still run.
        setToolCallLimit(2);
        vi.useFakeTimers({ toFake: ['Date'] });
        try {
            const chat = await newChat();
            state.toolPlan = [[['calculator', { expression: '1 + 1' }], ['calculator', { expression: '2 + 2' }], ['calculator', { expression: '3 + 3' }]]];
            const events = await send(chat.id, 'add these');
            expect(toolTurns(state.chatCalls[1]).map((turn) => turn.content)).toEqual(['2', '4', 'The limit on tool use for this hour has been reached. Answer without tools and say so.']);
            expect(events.filter((event) => event.type === 'toolFailed').map((event) => event.failed)).toEqual(['limited']);

            // Still inside the hour: refused.
            vi.setSystemTime(Date.now() + 59 * 60 * 1000);
            resetState();
            // A sign-in lasts less than an hour, so the person signs in again, as they would.
            signOut();
            await testStartup.loginAsUser('user');
            state.toolPlan = [[['calculator', { expression: '4 + 4' }]]];
            await send(chat.id, 'and this');
            expect(toolTurns(state.chatCalls[1])[0].content).toMatch(/^The limit on tool use for this hour/);

            // An hour after the first call, the count starts again.
            vi.setSystemTime(Date.now() + 2 * 60 * 1000);
            resetState();
            signOut();
            await testStartup.loginAsUser('user');
            state.toolPlan = [[['calculator', { expression: '5 + 5' }]]];
            await send(chat.id, 'and now');
            expect(toolTurns(state.chatCalls[1])[0].content).toBe('10');
        } finally {
            vi.useRealTimers();
            setToolCallLimit(null);
        }
    });

    test('an answer far too long is cut to the limit, and a call that fails or times out is said plainly while the reply goes on', async () => {
        const server = await addServer();
        const chat = await newChat({ tools: [server.id] });
        state.toolPlan = [[['notes_find_page', { words: 'huge' }]], [['notes_find_page', { words: 'slow' }]]];
        const events = await send(chat.id, 'find everything');
        const long = toolTurns(state.chatCalls[1])[0].content;
        expect(long.length).toBeLessThan(20100);
        expect(long.startsWith('x'.repeat(1000))).toBe(true);
        expect(events.filter((event) => event.type === 'toolFailed')).toHaveLength(1);
        expect(toolTurns(state.chatCalls[2])[1].content).toMatch(/\S/);
        expect(events.find((event) => event.type === 'done').reply.steps.map((step) => step.failed)).toEqual([null, expect.any(String)]);
        expect(events.find((event) => event.type === 'done').reply.steps[0].answer.length).toBeLessThanOrEqual(1503);
    });

    test('a tool from a tool server is offered under its own name, called on that server, and a failure is passed on plainly', async () => {
        const server = await addServer();
        const chat = await newChat({ tools: [server.id] });
        state.toolPlan = [[['notes_find_page', { words: 'launch' }]], [['notes_find_page', '{"words":"broken"}']]];
        const events = await send(chat.id, 'find the launch plan');
        expect(state.chatCalls[0].tools.map((tool) => tool.function.name)).toEqual(['notes_find_page', 'notes_write_page']);
        expect(called).toEqual([{ name: 'find_page', arguments: { words: 'launch' } }, { name: 'find_page', arguments: { words: 'broken' } }]);
        expect(events.filter((event) => event.type === 'tool').map((event) => event.tool)).toEqual(['Notes: Find a page', 'Notes: Find a page']);
        expect(toolTurns(state.chatCalls[1])[0].content).toBe('Found: the launch plan.');
        expect(toolTurns(state.chatCalls[2])[1].content).toBe('No page has those words.');
        expect(events.filter((event) => event.type === 'toolFailed').map((event) => event.failed)).toEqual(['failed']);
        expect(events.some((event) => event.type === 'permission')).toBe(false);
    });

    test('a tool that changes something waits for the person: allowed once it runs, refused it does not, and always stops the asking', async () => {
        const server = await addServer();
        const chat = await newChat({ tools: [server.id] });
        const turn = async (answer) => {
            state.toolPlan = [[['notes_write_page', { text: 'hello' }]]];
            state.chatCalls = [];
            const pending = send(chat.id, 'save this');
            let asked = false;
            if (answer) {
                asked = await until(() => waitingPermissions().some((waiting) => waiting.chatId === chat.id));
                if (asked) expect((await post(`/chats/${chat.id}/permission`, { id: waitingPermissions().find((waiting) => waiting.chatId === chat.id).id, answer })).status).toBe(200);
            }
            return { events: await pending, asked };
        };

        const once = await turn('once');
        expect(once.asked).toBe(true);
        expect(once.events.find((event) => event.type === 'permission')).toMatchObject({ tool: 'Notes: write_page', input: '{"text":"hello"}' });
        expect(called).toEqual([{ name: 'write_page', arguments: { text: 'hello' } }]);

        called.length = 0;
        const refused = await turn('no');
        expect(called).toEqual([]);
        expect(toolTurns(state.chatCalls[1])[0].content).toMatch(/did not allow/);
        expect(refused.events.find((event) => event.type === 'toolFailed')).toMatchObject({ failed: 'refused' });

        await turn('always');
        expect(called.length).toBe(1);
        const after = await turn(null);
        expect(after.events.some((event) => event.type === 'permission')).toBe(false);
        expect(called.length).toBe(2);
        expect(waitingPermissions()).toEqual([]);
    });

    test('only the person whose chat it is can answer, and a question that is not waiting cannot be answered', async () => {
        const server = await addServer();
        const chat = await newChat({ tools: [server.id] });
        state.toolPlan = [[['notes_write_page', { text: 'hello' }]]];
        const pending = send(chat.id, 'save this');
        expect(await until(() => waitingPermissions().length > 0)).toBe(true);
        const { id } = waitingPermissions()[0];
        expect((await post(`/chats/${chat.id}/permission`, { id: 'abcdef123456', answer: 'once' })).status).toBe(404);
        expect((await post(`/chats/${chat.id}/permission`, { id, answer: 'maybe' })).status).toBe(400);
        expect((await post(`/chats/${chat.id}/permission`, { id, answer: 'no' })).status).toBe(200);
        await pending;
        expect((await post(`/chats/${chat.id}/permission`, { id, answer: 'once' })).status).toBe(404);
        expect(called).toEqual([]);
    });

    test('a tool server that is not ready, or belongs to someone else, gives a chat no tools', async () => {
        const server = await addServer();
        await ToolServer.updateOne({ _id: server.id }, { health: 'unreachable' });
        const chat = await newChat({ tools: [server.id] });
        await send(chat.id);
        expect(state.chatCalls[0].tools).toBeUndefined();

        await ToolServer.updateOne({ _id: server.id }, { health: 'healthy' });
        signOut();
        await testStartup.loginAsUser('creator');
        resetState();
        const theirs = await newChat({ tools: [server.id, 'calculator'] });
        await send(theirs.id);
        expect(state.chatCalls[0].tools).toBeUndefined();
    });
});
