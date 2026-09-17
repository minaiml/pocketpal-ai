import {fetchServerProps} from '../props';
import {
  errorSlotsBareRouter,
  propsModelDescribing,
  propsRouterPlaceholder,
} from '../../../../jest/fixtures/llamaServerWire';

describe('fetchServerProps', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('parses contextLength from default_generation_settings.n_ctx and vision', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          default_generation_settings: {n_ctx: 4096},
          modalities: {vision: true, audio: false},
        }),
    });

    const {caps} = await fetchServerProps('http://localhost:8080');

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8080/props',
      expect.objectContaining({method: 'GET'}),
    );
    expect(caps).toEqual({
      contextLength: 4096,
      supportsVision: true,
      supportsAudio: false,
    });
  });

  it('falls back to top-level n_ctx and reports vision false when not present', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({n_ctx: 8192}),
    });

    const {caps} = await fetchServerProps('http://localhost:8080');
    expect(caps).toEqual({
      contextLength: 8192,
      supportsVision: false,
      supportsAudio: false,
    });
  });

  it('returns supportsVision false (defined, not omitted) when vision is off', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          default_generation_settings: {n_ctx: 4096},
          modalities: {vision: false},
        }),
    });

    const {caps} = await fetchServerProps('http://localhost:8080');
    // Defined false on a 2xx probe clears a stale true on a model swap; a
    // failed probe would resolve every tier to unknown instead.
    expect(caps).toEqual({
      contextLength: 4096,
      supportsVision: false,
      supportsAudio: false,
    });
  });

  it('resolves every tier to unknown on a non-2xx response', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: () => Promise.resolve(errorSlotsBareRouter),
    });

    await expect(fetchServerProps('http://localhost:8080')).resolves.toEqual({
      caps: {},
      props: {},
    });
  });

  it('resolves every tier to unknown on a network error (never throws)', async () => {
    global.fetch = jest.fn().mockRejectedValueOnce(new Error('boom'));

    await expect(fetchServerProps('http://localhost:8080')).resolves.toEqual({
      caps: {},
      props: {},
    });
  });

  it('resolves every tier to unknown on malformed JSON', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.reject(new Error('bad json')),
    });

    await expect(fetchServerProps('http://localhost:8080')).resolves.toEqual({
      caps: {},
      props: {},
    });
  });

  it('scopes the request to a model id when one is supplied', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          model_path: '/models/gemma-4-e2b.gguf',
          default_generation_settings: {n_ctx: 8192},
          modalities: {vision: true, video: true, audio: true},
        }),
    });

    const {caps} = await fetchServerProps(
      'http://localhost:8080',
      undefined,
      undefined,
      'gemma-4-e2b',
    );

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8080/props?model=gemma-4-e2b',
      expect.objectContaining({method: 'GET'}),
    );
    expect(caps).toEqual({
      contextLength: 8192,
      supportsVision: true,
      supportsAudio: true,
    });
  });

  it('url-encodes a model id containing a slash', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({n_ctx: 4096}),
    });

    await fetchServerProps(
      'http://localhost:8080',
      undefined,
      undefined,
      'unsloth/gemma-3-4b',
    );

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8080/props?model=unsloth%2Fgemma-3-4b',
      expect.objectContaining({method: 'GET'}),
    );
  });

  it('resolves nothing at all for a router placeholder body', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(propsRouterPlaceholder),
    });

    // n_ctx 0 is "unknown", not a window, and nothing about a model is
    // decidable from a body that describes none. The router's own sleep state
    // is not this model's, so presence stays unknown too.
    await expect(fetchServerProps('http://localhost:8080')).resolves.toEqual({
      caps: {},
      props: {},
    });
  });

  it('omits contextLength when n_ctx is zero, negative, or not a number', async () => {
    for (const n_ctx of [0, -1, NaN, '4096']) {
      global.fetch = jest.fn().mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            model_path: '/models/m.gguf',
            n_ctx,
            modalities: {},
          }),
      });

      const {caps} = await fetchServerProps('http://localhost:8080');
      expect(caps.contextLength).toBeUndefined();
    }
  });

  it('reports vision false when a real model body carries no modalities key', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          model_path: '/models/old-build.gguf',
          default_generation_settings: {n_ctx: 2048},
        }),
    });

    const {caps} = await fetchServerProps('http://localhost:8080');
    expect(caps).toEqual({
      contextLength: 2048,
      supportsVision: false,
      supportsAudio: false,
    });
  });

  it('decides vision from model_path alone when no window is reported', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          model_path: '/models/m.gguf',
          n_ctx: 0,
          modalities: {vision: true},
        }),
    });

    const {caps} = await fetchServerProps('http://localhost:8080');
    expect(caps).toEqual({supportsVision: true, supportsAudio: false});
  });

  it('splits a full model-describing body across the three tiers', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(propsModelDescribing),
    });

    const {caps, props, presence} = await fetchServerProps(
      'http://localhost:8080',
    );

    expect(caps).toEqual({
      contextLength: propsModelDescribing.default_generation_settings.n_ctx,
      supportsVision: false,
      supportsAudio: false,
    });

    // Projected from the capture rather than restated, so the four renames are
    // checked against the server's own spelling and not against themselves.
    const wire = propsModelDescribing.default_generation_settings.params;
    expect(props.samplerDefaults).toEqual({
      temperature: wire.temperature,
      top_p: wire.top_p,
      top_k: wire.top_k,
      min_p: wire.min_p,
      typical_p: wire.typical_p,
      xtc_threshold: wire.xtc_threshold,
      xtc_probability: wire.xtc_probability,
      penalty_last_n: wire.repeat_last_n,
      penalty_repeat: wire.repeat_penalty,
      penalty_freq: wire.frequency_penalty,
      penalty_present: wire.presence_penalty,
      mirostat: wire.mirostat,
      mirostat_tau: wire.mirostat_tau,
      mirostat_eta: wire.mirostat_eta,
      n_predict: wire.n_predict,
      n_probs: wire.n_probs,
    });
    expect(props.slotCount).toBe(propsModelDescribing.total_slots);
    // The wire carries both; nothing reads them, so nothing persists them.
    expect(propsModelDescribing.build_info).toBeTruthy();
    expect(propsModelDescribing.model_alias).toBeTruthy();
    expect(props).not.toHaveProperty('buildInfo');
    expect(props).not.toHaveProperty('modelAlias');

    expect(presence).toEqual({
      isSleeping: propsModelDescribing.is_sleeping,
      probedUrl: 'http://localhost:8080',
      at: expect.any(Number),
    });
  });

  // The read names are derived from the llama.cpp send map, so a param read
  // under its send name would be silently wrong wherever the two differ.
  it('reads each default under the name the server reports, not the send name', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(propsModelDescribing),
    });

    const {props} = await fetchServerProps('http://localhost:8080');

    // n_predict is the one param the server reports under a different name
    // than the one we send it as (max_completion_tokens).
    expect(props?.samplerDefaults?.n_predict).toBe(-1);
    expect(props?.samplerDefaults?.n_probs).toBe(0);
    // The server reports its live seed, which is not a default worth keeping.
    expect(props?.samplerDefaults).not.toHaveProperty('seed');
  });

  it('keeps a server default that is zero', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(propsModelDescribing),
    });

    const {props} = await fetchServerProps('http://localhost:8080');

    // Five of the captured defaults are legitimately 0. A parser that treated
    // absent and zero alike would drop exactly these, leaving the indicator
    // silently dead on five controls and working on the rest.
    const wire = propsModelDescribing.default_generation_settings.params;
    expect([
      wire.xtc_probability,
      wire.mirostat,
      wire.frequency_penalty,
      wire.presence_penalty,
    ]).toEqual([0, 0, 0, 0]);
    expect(props.samplerDefaults?.xtc_probability).toBe(0);
    expect(props.samplerDefaults?.mirostat).toBe(0);
    expect(props.samplerDefaults?.penalty_freq).toBe(0);
    expect(props.samplerDefaults?.penalty_present).toBe(0);
  });

  it('drops a sampler default the server reports as a non-number', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          ...propsModelDescribing,
          default_generation_settings: {
            ...propsModelDescribing.default_generation_settings,
            params: {
              ...propsModelDescribing.default_generation_settings.params,
              temperature: 'hot',
              top_k: Infinity,
              min_p: null,
              top_p: NaN,
            },
          },
        }),
    });

    const {props} = await fetchServerProps('http://localhost:8080');

    // Absent beats wrong: a control whose default cannot be read offers no
    // reset rather than one that would send a value the server never gave.
    expect(props.samplerDefaults).not.toHaveProperty('temperature');
    expect(props.samplerDefaults).not.toHaveProperty('top_k');
    expect(props.samplerDefaults).not.toHaveProperty('min_p');
    expect(props.samplerDefaults).not.toHaveProperty('top_p');
    // The rest of the body still parses.
    expect(props.samplerDefaults?.mirostat).toBe(0);
  });

  it('does not offer the live seed as a default to return to', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(propsModelDescribing),
    });

    const {props} = await fetchServerProps('http://localhost:8080');

    expect(propsModelDescribing.default_generation_settings.params.seed).toBe(
      4294967295,
    );
    expect(props.samplerDefaults).not.toHaveProperty('seed');
  });

  it('leaves a chat-template flag this build does not report unknown', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(propsModelDescribing),
    });

    const {props} = await fetchServerProps('http://localhost:8080');

    expect(propsModelDescribing.chat_template_caps).not.toHaveProperty(
      'supports_thinking',
    );
    expect(props.chatTemplateCaps).toEqual({supportsTools: true});
    expect(props.chatTemplateCaps).not.toHaveProperty('supportsThinking');
  });

  it('records a chat-template flag a newer build does report', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          ...propsModelDescribing,
          chat_template_caps: {
            ...propsModelDescribing.chat_template_caps,
            supports_thinking: true,
          },
        }),
    });

    const {props} = await fetchServerProps('http://localhost:8080');

    expect(props.chatTemplateCaps).toEqual({
      supportsTools: true,
      supportsThinking: true,
    });
  });

  it('omits a slot count that is not a positive whole number', async () => {
    for (const total_slots of [0, -1, 2.5, '4', null]) {
      global.fetch = jest.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({...propsModelDescribing, total_slots}),
      });

      const {props} = await fetchServerProps('http://localhost:8080');
      expect(props.slotCount).toBeUndefined();
    }
  });

  it('reads sleep state from a body that describes nothing else', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({is_sleeping: true}),
    });

    // A sleeping child may report almost nothing, which is exactly when the
    // observation matters. The other two tiers stay unknown.
    await expect(fetchServerProps('http://localhost:8080')).resolves.toEqual({
      caps: {},
      props: {},
      presence: {
        isSleeping: true,
        probedUrl: 'http://localhost:8080',
        at: expect.any(Number),
      },
    });
  });

  it('reads no descriptive field from a body that describes no model', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          model_path: 'none',
          build_info: 'b9976-e3546c794',
          total_slots: 4,
        }),
    });

    await expect(fetchServerProps('http://localhost:8080')).resolves.toEqual({
      caps: {},
      props: {},
    });
  });

  it('records the url it asked, so a reader can tell what it describes', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(propsModelDescribing),
    });

    const {presence} = await fetchServerProps('http://192.168.1.100:8080/');

    expect(presence?.probedUrl).toBe('http://192.168.1.100:8080/');
  });

  it('bounds an omitted timeout at the props default, not the connection default', async () => {
    jest.useFakeTimers();
    const setTimeoutSpy = jest.spyOn(global, 'setTimeout');
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({}),
    });

    await fetchServerProps('http://localhost:8080');

    expect(setTimeoutSpy).toHaveBeenCalledWith(expect.any(Function), 5000);
    setTimeoutSpy.mockRestore();
    jest.useRealTimers();
  });

  it('falls back to the props default when the server timeout is unusable', async () => {
    jest.useFakeTimers();
    const setTimeoutSpy = jest.spyOn(global, 'setTimeout');
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    });

    for (const timeout of [0, -1, NaN]) {
      setTimeoutSpy.mockClear();
      await fetchServerProps('http://localhost:8080', undefined, timeout);
      expect(setTimeoutSpy).toHaveBeenCalledWith(expect.any(Function), 5000);
    }

    setTimeoutSpy.mockClear();
    await fetchServerProps('http://localhost:8080', undefined, 20000);
    expect(setTimeoutSpy).toHaveBeenCalledWith(expect.any(Function), 20000);

    setTimeoutSpy.mockRestore();
    jest.useRealTimers();
  });
});
