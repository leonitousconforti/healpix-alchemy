# Changelog fragments

This directory holds [towncrier](https://towncrier.readthedocs.io/) news
fragments: one small Markdown file per change, compiled into `CHANGES.md` at
release time. This avoids merge conflicts in the changelog and makes each pull
request document its own change.

## Adding a fragment

Create a file named `<pr-number>.<type>.md` (for example `123.bugfix.md`), or
`+<slug>.<type>.md` (for example `+fix-overflow.bugfix.md`) if there is no pull
request number yet. The file contains a short, user-facing description of the
change written in the past-tense-free style of the existing changelog.

You can also run:

    towncrier create <pr-number>.<type>.md

Valid types:

- `breaking`: breaking changes
- `feature`: new features
- `bugfix`: bug fixes
- `misc`: everything else (dependency bumps, docs, internal changes)

Changes that do not affect users (for example CI tweaks) can skip the fragment
requirement by adding the `skip-changelog` label to the pull request.

## Releasing

    towncrier build --version <X.Y.Z>

This inserts a new section into `CHANGES.md`, deletes the consumed fragments,
and stages everything for commit. Then commit, tag `v<X.Y.Z>` (the version
comes from the git tag via setuptools_scm), and create a GitHub release to
trigger the PyPI upload.

To preview the next release without changing anything:

    towncrier build --draft --version <X.Y.Z>
