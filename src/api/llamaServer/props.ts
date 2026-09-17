import {
  RemoteModelCaps,
  RemoteModelPresence,
  RemoteModelProps,
  SamplerDefaults,
} from '../../utils/types';
import {finiteNumber} from '../../utils/finite';
import type {SamplerParam} from '../../utils/samplerParams';
import {buildHeaders, normalizeUrl, resolveTimeout} from '../http';
import {llamaCpp} from '../servers/llamaCpp';

/**
 * The name each control is reported under, derived from the names it is sent
 * under so the two cannot drift. `n_predict` is the one entry whose read name
 * is not its send name: it is reported under `n_predict` and sent as
 * `max_completion_tokens`.
 *
 * Total over `SamplerParam`, so a control added to the vocabulary without a
 * llama.cpp name fails to compile here.
 */
export const PROPS_READ_NAMES = {
  ...llamaCpp.sendNames,
  n_predict: 'n_predict',
} satisfies Record<SamplerParam, string>;

// A fire-and-forget probe must neither inherit the 30 s connection default nor
// an arbitrarily large user-set timeout.
export const PROPS_TIMEOUT_MS = 5000;

/** One `/props` response, split by how long each fact stays true. */
export interface ServerPropsResult {
  caps: RemoteModelCaps;
  props: RemoteModelProps;
  presence?: RemoteModelPresence;
}

const definiteBoolean = (value: unknown): boolean | undefined =>
  typeof value === 'boolean' ? value : undefined;

/**
 * The server's own generation defaults, read under the same wire names a
 * request is sent under. Values sit under `params` on current builds and
 * directly on `default_generation_settings` on older ones.
 *
 * `seed` is skipped: the server reports the live seed, which is not a value
 * anyone should be offered as a default to return to.
 */
function readSamplerDefaults(generationSettings: any): SamplerDefaults {
  const defaults: SamplerDefaults = {};
  for (const param of Object.keys(PROPS_READ_NAMES) as SamplerParam[]) {
    if (param === 'seed') {
      continue;
    }
    const wireName = PROPS_READ_NAMES[param];
    const value = finiteNumber(
      generationSettings?.params?.[wireName] ?? generationSettings?.[wireName],
    );
    if (value !== undefined) {
      defaults[param] = value;
    }
  }
  return defaults;
}

/**
 * The two chat-template flags this app has a use for. `supports_thinking`
 * exists only on builds newer than b9976, so an absent key is unknown rather
 * than a definite `false`.
 */
function readChatTemplateCaps(
  wire: any,
): RemoteModelProps['chatTemplateCaps'] | undefined {
  const supportsTools = definiteBoolean(wire?.supports_tools);
  const supportsThinking = definiteBoolean(wire?.supports_thinking);
  if (supportsTools === undefined && supportsThinking === undefined) {
    return undefined;
  }
  return {
    ...(supportsTools !== undefined ? {supportsTools} : {}),
    ...(supportsThinking !== undefined ? {supportsThinking} : {}),
  };
}

/**
 * Fetch what a llama.cpp server reports for one model via GET /props.
 * Pure: parses the response into the three tiers and never throws — a timeout,
 * non-2xx, or malformed body resolves every tier to unknown so the caller's
 * models path and connection are never affected. `/props` is
 * llama.cpp-specific; callers gate on serverType before invoking.
 *
 * `modelId` scopes the request (`?model=<id>`). A multi-model router answers
 * the bare form with a placeholder (`role: 'router'`, `model_path: 'none'`,
 * `n_ctx: 0`, `modalities` absent) that describes no model, so a field is only
 * ever returned when the body describes an actually loaded model. Absent field
 * = unknown; the caller merges field-wise and never blanks a known value.
 *
 * Sleep state takes the weaker gate: a sleeping child may report almost
 * nothing else, so requiring a model-describing body would suppress the very
 * observation the field exists for.
 *
 * Key names verified against live llama.cpp builds (b9910, b9976): context
 * window is `default_generation_settings.n_ctx` (top-level `n_ctx` is an
 * older-build fallback); vision is `modalities.vision`.
 */
export async function fetchServerProps(
  serverUrl: string,
  apiKey?: string,
  timeoutMs?: number,
  modelId?: string,
): Promise<ServerPropsResult> {
  const url =
    `${normalizeUrl(serverUrl)}/props` +
    (modelId ? `?model=${encodeURIComponent(modelId)}` : '');
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    resolveTimeout(timeoutMs, PROPS_TIMEOUT_MS),
  );

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: buildHeaders(apiKey),
      signal: controller.signal,
    });
    if (!response.ok) {
      return {caps: {}, props: {}};
    }
    const data = await response.json();
    const caps: RemoteModelCaps = {};
    const props: RemoteModelProps = {};
    let presence: RemoteModelPresence | undefined;

    const nCtx: unknown =
      data?.default_generation_settings?.n_ctx ?? data?.n_ctx;
    if (typeof nCtx === 'number' && Number.isFinite(nCtx) && nCtx > 0) {
      caps.contextLength = nCtx;
    }

    const modelPath: unknown = data?.model_path;
    const describesModel =
      (typeof modelPath === 'string' &&
        modelPath !== '' &&
        modelPath !== 'none') ||
      caps.contextLength !== undefined;
    const isRouterPlaceholder =
      data?.role === 'router' ||
      (modelPath === 'none' && data?.modalities === undefined);

    if (describesModel) {
      caps.supportsVision = data?.modalities?.vision === true;
      caps.supportsAudio = data?.modalities?.audio === true;

      const samplerDefaults = readSamplerDefaults(
        data?.default_generation_settings,
      );
      if (Object.keys(samplerDefaults).length > 0) {
        props.samplerDefaults = samplerDefaults;
      }

      const slotCount = finiteNumber(data?.total_slots);
      if (
        slotCount !== undefined &&
        Number.isInteger(slotCount) &&
        slotCount > 0
      ) {
        props.slotCount = slotCount;
      }

      const chatTemplateCaps = readChatTemplateCaps(data?.chat_template_caps);
      if (chatTemplateCaps !== undefined) {
        props.chatTemplateCaps = chatTemplateCaps;
      }
    }

    if (!isRouterPlaceholder) {
      const isSleeping = definiteBoolean(data?.is_sleeping);
      if (isSleeping !== undefined) {
        presence = {isSleeping, probedUrl: serverUrl, at: Date.now()};
      }
    }

    return {caps, props, presence};
  } catch {
    return {caps: {}, props: {}};
  } finally {
    clearTimeout(timeout);
  }
}
