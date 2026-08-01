# R2 Static-Literal Boundary Decision v1

## Corrected Normal Output Audit

The format-preserving Normal Curated output contains 7,271 chunks and
2,568,426 whitespace-proxy tokens. The diagnostic one/two-line surface is 25
chunks and 13,055 proxy tokens, approximately 0.51% of the Curated proxy
token total. It is not a large hidden compression reservoir.

## Observed Forms

The 25 chunks are Python static payload forms, including a serialized protobuf
descriptor, a puzzle-input literal, a numeric matrix, metadata string tables,
and a large embedded daily-data table. The generated protobuf record is
already excluded by the active generated-artifact policy before the Curated
audit; the remaining forms include ordinary static literals and data fixtures.

## Decision

No `static_literal`, `one_line`, `minified_like`, `large_table`, or
`data_dump` default deletion rule is introduced.

Those surface properties do not prove the absence of LM-learning payload. A
static literal can be a useful API contract, a test fixture, a domain example,
or part of an implementation a model must learn to read and write. A threshold
on line count, character count, literal ratio, or apparent information density
would be an unvalidated proxy and is prohibited from the active runtime.

The inspection used raw-record path metadata only as an audit sidecar to
interpret false-positive risk. The runtime did not and must not read it.

## Consequence

The repeated-label-block rule is retired: it had zero spans in corrected Code,
Math, and General development corpora. The static-literal category is also
archived as a discovery outcome, not promoted. Neither changes Normal or Hard.

For materially stronger compression, the next experiment must use a corpus
with demonstrably more explicit removable artifacts or establish a new bounded
rule with independent evidence. It cannot obtain a target deletion rate by
widening text-shape heuristics.
