# Legacy Removal Decision

## Current State

Historical temporal-code implementations for numbered commands `76` through
`135` live in `archive/temporal_code`. Root commands are compatibility wrappers.

The active canonical registry does not execute these historical commands.

## Actual Reduction Boundary

Removing the archived implementations and their root wrappers would remove:

- 60 archived implementation files, about 7,662 lines
- 60 root compatibility wrappers, about 480 lines
- total removable code surface: about 8,142 lines

## What Physical Removal Changes

Physical removal preserves the active canonical framework path, but removes:

- documented historical numbered commands;
- tests that reproduce historical temporal-code collection/Stage-B operations;
- the ability to rebuild or inspect those historical artifacts from this branch.

It does not remove current framework evidence outputs by itself. It does make
the `full` historical validation scope and the compatibility manifest obsolete;
those must be removed or rewritten in the same change.

## Required Decision

Keep this branch as a reproducibility-complete research record, or create a
lean active-framework branch that removes this historical implementation set.
