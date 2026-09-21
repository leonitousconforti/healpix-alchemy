# Contributing

## Changesets

Releases are managed by [Knope](https://knope.tech). Every pull request that
affects users must include a changeset: a small Markdown file in `.changeset/`
that describes the change and how it should bump the version. Changesets avoid
merge conflicts in the changelog and make each pull request document its own
change.

Create one by running `knope document-change`, or write the file by hand:

```markdown
---
default: patch
---

# Short, user-facing summary of the change

Optional longer description in the past-tense-free style of the existing
changelog.
```

Valid change types:

- `major`: breaking changes
- `minor`: new features
- `patch`: bug fixes
- `misc`: everything else (dependency bumps, docs, internal changes); bumps the
  patch version and appears under "Other changes" in the changelog

Changes that do not affect users (for example CI tweaks) can skip the changeset
requirement by adding the `skip-changelog` label to the pull request.

## Releasing

On every push to `main`, the `prepare-release` workflow runs
`knope prepare-release`, which opens (or updates) a release PR from the
`release` branch. That PR consumes the pending changesets, computes the next
version, and inserts the new section into `CHANGES.md`.

Merging the release PR runs `knope release`, which tags `v<X.Y.Z>` (the package
version comes from the git tag via setuptools_scm) and creates a GitHub
release, and then uploads the package to PyPI.
