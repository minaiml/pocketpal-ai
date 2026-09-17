/**
 * Pins what the remote request path does today, across every server type, so a
 * restructure can be checked against it. Only the adapters in the first block
 * may change when the structure does; a change anywhere below is a behaviour
 * change and has to be intended.
 */
import {OpenAICompletionEngine} from '../completionEngines';
import {fetchServerProps} from '../llamaServer/props';
import {deriveListCaps} from '../servers/listCaps';
import {EFFORT_LEVELS} from '../../utils/reasoningCapability';
import {
  propsModelDescribing,
  propsRouterPlaceholder,
  streamFinishChunk,
} from '../../../jest/fixtures/llamaServerWire';
import {cacheReuseTimings} from '../../../jest/fixtures/llamaServerTimings';
import {
  directTextModelsBody,
  directVisionModelsBody,
  routerModelsBody,
} from '../../../jest/fixtures/remoteModelList';

// ---- Adapters: the only part a restructure is allowed to edit. ----

const engineFor = (serverType: unknown) =>
  new OpenAICompletionEngine({
    url: 'http://localhost:8080',
    remoteModelId: 'm',
    serverType: serverType as any,
  });

const readProps = (modelId?: string) =>
  fetchServerProps('http://localhost:8080', undefined, undefined, modelId);

const listCapsFor = (row: any, serverType: unknown) =>
  deriveListCaps(row, serverType as any);

// ---- Harness ----

class FakeXHR {
  static last: FakeXHR;
  body = '';
  url = '';
  status = 0;
  readyState = 0;
  statusText = '';
  responseText = '';
  onreadystatechange: (() => void) | null = null;
  onprogress: (() => void) | null = null;
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  constructor() {
    FakeXHR.last = this;
  }
  open(_method: string, url: string) {
    this.url = url;
  }
  setRequestHeader() {}
  send(body?: string) {
    this.body = body ?? '';
  }
  abort() {
    this.onabort?.();
  }
  finish(frames: object[]) {
    this.readyState = 2;
    this.status = 200;
    this.onreadystatechange?.();
    this.responseText = frames
      .map(f => `data: ${JSON.stringify(f)}\n\n`)
      .concat('data: [DONE]\n\n')
      .join('');
    this.onprogress?.();
    this.readyState = 4;
    this.onload?.();
  }
}

const STOP_FRAME = {choices: [{delta: {}, finish_reason: 'stop'}]};

/** Runs one turn and returns the request body and the engine's result. */
const turn = async (
  serverType: unknown,
  params: Record<string, unknown>,
  frames: object[] = [STOP_FRAME],
) => {
  const pending = engineFor(serverType).completion({
    messages: [{role: 'user', content: 'Hi'}],
    ...params,
  } as any);
  const xhr = FakeXHR.last;
  const body = JSON.parse(xhr.body);
  xhr.finish(frames);
  return {body, result: await pending};
};

let realXHR: typeof XMLHttpRequest;
beforeEach(() => {
  realXHR = global.XMLHttpRequest;
  (global as any).XMLHttpRequest = FakeXHR;
});
afterEach(() => {
  global.XMLHttpRequest = realXHR;
});

// ---- Axes ----

const SERVER_TYPES: [string, unknown][] = [
  ['llama.cpp', 'llama.cpp'],
  ['LM Studio', 'LM Studio'],
  ['Ollama', 'Ollama'],
  ['OpenAI', 'OpenAI'],
  ['vLLM', 'vLLM'],
  ['unknown', 'unknown'],
  ['empty string', ''],
  ['undefined', undefined],
  ['case variant', 'LLAMA.CPP'],
  ['free string', 'my server'],
];

const REASONING: [string, unknown][] = [
  ['no intent', undefined],
  ['off', {enabled: false}],
  ['off with effort', {enabled: false, effort: 'high'}],
  ['on, no effort', {enabled: true}],
  ...EFFORT_LEVELS.map((e): [string, unknown] => [
    `on, ${e}`,
    {enabled: true, effort: e},
  ]),
  ['on, not a level', {enabled: true, effort: 'extreme'}],
];

const SAMPLERS: [string, Record<string, unknown>][] = [
  ['none', {}],
  ['temperature only', {temperature: 0.2}],
  [
    'distinctive',
    {
      temperature: 0.33,
      top_p: 0.77,
      n_predict: 256,
      top_k: 11,
      min_p: 0.11,
      typical_p: 0.91,
      xtc_threshold: 0.31,
      xtc_probability: 0.21,
      penalty_last_n: 41,
      penalty_repeat: 1.11,
      penalty_freq: 0.41,
      penalty_present: 0.51,
      mirostat: 2,
      mirostat_tau: 4.1,
      mirostat_eta: 0.21,
      seed: 12345,
      n_probs: 2,
      stop: ['</s>'],
    },
  ],
  ['unlimited length', {n_predict: -1, top_k: 40}],
  [
    'non-finite values',
    {
      temperature: Number.NaN,
      top_p: Infinity,
      n_predict: Number.NaN,
      top_k: Number.NaN,
      min_p: Infinity,
    },
  ],
];

// ---- Request bodies ----

describe('request body', () => {
  describe.each(SERVER_TYPES)('server type: %s', (_label, type) => {
    it.each(REASONING)('reasoning: %s', async (_r, reasoning) => {
      const {body} = await turn(type, {reasoning});
      expect(body).toMatchSnapshot();
    });

    it.each(SAMPLERS)('samplers: %s', async (_s, samplers) => {
      const {body} = await turn(type, samplers);
      expect(body).toMatchSnapshot();
    });

    it('samplers and reasoning together', async () => {
      const {body} = await turn(type, {
        ...SAMPLERS[2][1],
        reasoning: {enabled: true, effort: 'high'},
      });
      expect(body).toMatchSnapshot();
    });

    it('reasoning off with samplers', async () => {
      const {body} = await turn(type, {
        ...SAMPLERS[2][1],
        reasoning: {enabled: false},
      });
      expect(body).toMatchSnapshot();
    });
  });
});

// ---- Response reading ----

const withoutText = ({text: _text, content: _content, ...rest}: any) => rest;

const FINISH_FRAMES: [string, object[]][] = [
  ['no timings', [STOP_FRAME]],
  ['captured finish chunk', [streamFinishChunk]],
  ['cache reuse timings', [{...STOP_FRAME, timings: cacheReuseTimings}]],
  [
    'malformed timings',
    [
      {
        ...STOP_FRAME,
        timings: {
          ...streamFinishChunk.timings,
          prompt_per_second: 'NaN',
          predicted_per_second: null,
          cache_n: '15',
          predicted_n: Number.NaN,
        },
      },
    ],
  ],
  ['length stop', [{choices: [{delta: {}, finish_reason: 'length'}]}]],
];

describe('response reading', () => {
  describe.each(SERVER_TYPES)('server type: %s', (_label, type) => {
    it.each(FINISH_FRAMES)('%s', async (_f, frames) => {
      const {result} = await turn(type, {}, frames);
      expect(withoutText(result)).toMatchSnapshot();
    });
  });
});

// ---- /props ----

describe('/props parse', () => {
  const realFetch = global.fetch;
  // The presence tier stamps an observation time.
  beforeEach(() => {
    jest.spyOn(Date, 'now').mockReturnValue(1_700_000_000_000);
  });
  afterEach(() => {
    global.fetch = realFetch;
    jest.restoreAllMocks();
  });

  const respond = (init: {ok?: boolean; status?: number; json?: unknown}) => {
    const calls: string[] = [];
    global.fetch = jest.fn(async (url: string) => {
      calls.push(url);
      return {
        ok: init.ok ?? true,
        status: init.status ?? 200,
        json: async () => {
          if (init.json instanceof Error) {
            throw init.json;
          }
          return init.json;
        },
      };
    }) as any;
    return calls;
  };

  it.each([
    ['model-describing body', {json: propsModelDescribing}],
    ['router placeholder', {json: propsRouterPlaceholder}],
    ['server error', {ok: false, status: 500, json: {}}],
    ['unparseable body', {json: new Error('bad json')}],
    ['empty object', {json: {}}],
    ['null body', {json: null}],
    [
      'wrong-typed and non-finite values',
      {
        json: {
          ...propsModelDescribing,
          total_slots: Infinity,
          modalities: {vision: 'yes', audio: 1},
          chat_template_caps: {supports_tools: 'true'},
          default_generation_settings: {
            ...propsModelDescribing.default_generation_settings,
            n_ctx: Number.NaN,
            params: {
              ...propsModelDescribing.default_generation_settings.params,
              temperature: 'hot',
              top_k: Infinity,
              min_p: null,
              top_p: Number.NaN,
            },
          },
        },
      },
    ],
  ])('%s', async (_label, init) => {
    const scoped = respond(init);
    const scopedResult = await readProps(
      'bartowski/Qwen_Qwen3-1.7B-GGUF:Q4_K_M',
    );
    const bare = respond(init);
    const bareResult = await readProps();
    expect({
      scoped: {url: scoped, result: scopedResult},
      bare: {url: bare, result: bareResult},
    }).toMatchSnapshot();
  });

  it('a failed request', async () => {
    global.fetch = jest.fn(async () => {
      throw new Error('offline');
    }) as any;
    expect(await readProps('m')).toMatchSnapshot();
  });
});

// ---- List-row caps ----

describe('list-row caps', () => {
  const rows: [string, any][] = [
    ...routerModelsBody.data.map((r: any): [string, any] => [
      `router ${r.id}`,
      r,
    ]),
    ...directVisionModelsBody.data.map((r: any): [string, any] => [
      `direct vision ${r.id}`,
      r,
    ]),
    ...directTextModelsBody.data.map((r: any): [string, any] => [
      `direct text ${r.id}`,
      r,
    ]),
    ['absent row', undefined],
  ];

  describe.each(SERVER_TYPES)('server type: %s', (_label, type) => {
    it.each(rows)('%s', (_r, row) => {
      expect(listCapsFor(row, type)).toMatchSnapshot();
    });
  });
});
