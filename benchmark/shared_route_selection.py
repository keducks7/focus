"""Training-free route selection. Pure Python; no model or GPU dependency."""
import math

METHODS = ('joint', 'independent', 'joint_no_fixed')


def select_routes(ids, weights, compress, epsilon, method='joint'):
    """Return per-row keep flags and metrics; fixed rows always keep all routes.

    IDs are unique within each row. epsilon is omitted *normalized route mass*,
    not an output-error bound. Ties use expert ID for reproducibility.
    """
    if method not in METHODS or not 0 <= epsilon < 1:
        raise ValueError('Invalid method or epsilon (expected 0 <= epsilon < 1).')
    if not (len(ids) == len(weights) == len(compress)):
        raise ValueError('Mismatched row counts.')
    probs = []
    for es, ws in zip(ids, weights):
        if not es or len(es) != len(ws) or len(set(es)) != len(es):
            raise ValueError('Each route row must be nonempty with distinct IDs and matching weights.')
        if any(not math.isfinite(w) or w < 0 for w in ws) or sum(ws) <= 0:
            raise ValueError('Route weights must be finite, nonnegative and have positive total.')
        probs.append([w / sum(ws) for w in ws])
    target = 1 - epsilon
    fixed = {e for es, c in zip(ids, compress) if not c for e in es}
    support = set() if method == 'joint_no_fixed' else set(fixed)
    if epsilon and method != 'independent':
        while True:
            gains = {}
            for es, ps, c in zip(ids, probs, compress):
                if not c:
                    continue
                deficit = max(0., target - sum(p for e, p in zip(es, ps) if e in support))
                if deficit <= 1e-12:
                    continue
                for e, p in zip(es, ps):
                    if e not in support:
                        gains[e] = gains.get(e, 0.) + min(p, deficit)
            if not gains:
                break
            support.add(min(gains, key=lambda e: (-gains[e], e)))
    keep = []
    for es, ps, c in zip(ids, probs, compress):
        if not c or epsilon == 0:
            keep.append([True] * len(es))
            continue
        eligible = [j for j, e in enumerate(es) if method == 'independent' or e in support]
        chosen = [False] * len(es)
        mass = 0.
        for j in sorted(eligible, key=lambda j: (-ps[j], es[j])):
            chosen[j] = True
            mass += ps[j]
            if mass >= target - 1e-12:
                break
        if mass < target - 1e-12:
            raise RuntimeError('Coverage constraint not satisfied.')
        keep.append(chosen)
    before = {e for es in ids for e in es}
    after = {e for es, ks in zip(ids, keep) for e, k in zip(es, ks) if k}
    masses = [sum(p for p, k in zip(ps, ks) if k)
              for ps, ks, c in zip(probs, keep, compress) if c]
    metrics = dict(active_before=len(before), active_after=len(after), fixed_active=len(fixed),
                   assignments_before=sum(map(len, ids)), assignments_after=sum(sum(k) for k in keep),
                   compressed_assignments_before=sum(len(es) for es, c in zip(ids, compress) if c),
                   compressed_assignments_after=sum(sum(k) for k, c in zip(keep, compress) if c),
                   compressed_tokens=sum(compress), min_retained_mass=min(masses, default=1.),
                   mean_retained_mass=sum(masses)/len(masses) if masses else 1.)
    return keep, metrics


def reweight(weights, keep, renormalize=True):
    if all(keep):
        return list(weights)
    retained = sum(w for w, k in zip(weights, keep) if k)
    if retained <= 0:
        raise ValueError('Cannot remove all positive route mass.')
    scale = sum(weights) / retained if renormalize else 1.
    return [w * scale if k else 0. for w, k in zip(weights, keep)]
