# Changelog

Notable changes.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [2026.739679.1] - 2026-03-04
- Redefine stored data structures, dropping constructed arrays other than aggregate purchase probability

## [2026.739679.1] - 2026-03-04
- Workaround type-checker failure in Enameled constructor from yaml

## [2026.739679.0] - 2026-03-04

- Improved documentation
- Expanded PriceSpec

## [2026.739669.0] - 2026-02-22

- Parallelized data-sample generation function as a whole
- Lowered SUBSAMPLE_SIZE to reduce memory utilization and improve performance (avoids paging with 16G RAM, in most cases)
- Parallelized computation of UPP test-counts on existing dataset
- Collected margin resampling variants into a single function to better contrast against purely synthetic margin generation
- HMT market definition restriction now fully implemented
- Implemented reading the following constants from environment variables (env var name prefixed with "MERGERON_", e.g., "MERGERON_WORK_DIR"):
    - WORK_DIR
    - DEFAULT_REC
    - NTHREADS
    - SUBSAMPLE_SIZE
    - HSR_BETA_PARMS

## [2026.739659.2] - 2026-02-12

- Regenerate test data.

## [2026.739659.1] - 2026-02-12

- Fix errors in document generation and type-ing.

## [2026.739659.0] - 2026-02-12

### Added

- Draws from multimodal empirical margin distribution specified by `PCMDistribution.EMPR_M`.
- Draws from unimodal empirical margin distribution specified by `PCMDistribution.EMPR_U`.
- Market share specification can now include `lower_bound`, restricting the market-share sample.
- HSR filing test type which allows generation of a "test" random variable having Beta distribution with given parameters, `HSR_BETA_PARMS`.
- Data structures for (a) expanding options for specification of price distribution, and (b) restricting draws to those including a relevant antitrust market for the first merging firm. Include shares of all firms in the market-share data array, not just those of the merging firms.
- Integers PPN_S and PPN_L for seeding random number generators.

### Changed

- `PubYear` is a literal type, not a subclass of ``int``.
- Type, `SamplingFunctionKWArgs` is unused.

### Deprecated

- `PriceSpec`, presently defined as an `Enum`, to be redefined as a class to allow richer specification of distribution of prices and costs.
- Default for `ShareSpec.lower_bound`, presently $1\%$, is to be $0$.
