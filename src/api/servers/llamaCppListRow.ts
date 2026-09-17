import type {ListDerivedCaps, RemoteModelInfo} from '../../utils/types';

const CONTEXT_FLAGS = ['--ctx-size', '-c'];

const positiveInt = (raw: string | undefined): number | undefined => {
  if (raw === undefined || !/^\d+$/.test(raw)) {
    return undefined;
  }
  const value = parseInt(raw, 10);
  return Number.isSafeInteger(value) && value > 0 ? value : undefined;
};

/**
 * The launch window, from the last `--ctx-size`/`-c` in either the
 * `--ctx-size 8192` or `--ctx-size=8192` form. Last occurrence wins, matching
 * the server's own argument handling. No other argument is read.
 *
 * `--ctx-size 0` means "use the model's trained window", which is not a window
 * this can report, so it resolves to unknown like every other parse failure.
 */
const contextFromArgs = (args: unknown): number | undefined => {
  if (!Array.isArray(args)) {
    return undefined;
  }
  let found: number | undefined;
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (typeof arg !== 'string') {
      continue;
    }
    if (CONTEXT_FLAGS.includes(arg)) {
      found = positiveInt(
        typeof args[i + 1] === 'string' ? args[i + 1] : undefined,
      );
      continue;
    }
    const inline = CONTEXT_FLAGS.map(flag => `${flag}=`).find(prefix =>
      arg.startsWith(prefix),
    );
    if (inline) {
      found = positiveInt(arg.slice(inline.length));
    }
  }
  return found;
};

export function readLlamaCppListRow(
  row: RemoteModelInfo | undefined,
): ListDerivedCaps {
  const caps: ListDerivedCaps = {tier: 'list'};
  if (!row) {
    return caps;
  }

  const modalities = row.architecture?.input_modalities;
  if (Array.isArray(modalities)) {
    // The array always carries `text`, so a missing `image` is the server
    // answering "no", not failing to answer.
    caps.supportsVision = modalities.includes('image');
  } else if (Array.isArray(row.capabilities)) {
    caps.supportsVision = row.capabilities.includes('multimodal');
  }

  const reported =
    typeof row.meta?.n_ctx === 'number' && Number.isSafeInteger(row.meta.n_ctx)
      ? row.meta.n_ctx
      : undefined;
  // A loaded child's own report beats the command that launched it.
  const contextLength =
    (reported !== undefined && reported > 0 ? reported : undefined) ??
    contextFromArgs(row.status?.args);
  if (contextLength !== undefined) {
    caps.contextLength = contextLength;
  }

  return caps;
}
