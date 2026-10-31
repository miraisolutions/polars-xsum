#!/usr/bin/env bash
# Prints the CHANGELOG.md section of a release, i.e. the lines below the
# heading "## [VERSION] - YYYY-MM-DD" up to the next "## " heading.
#
# Fails if there is no such section or if it is empty, so that a release
# cannot be published without its changelog entry.
#
# Usage: release-notes.sh VERSION
set -euo pipefail

version=${1:?usage: release-notes.sh VERSION}
changelog="$(dirname "$0")/../../CHANGELOG.md"

notes=$(awk -v heading="## [${version}] - " '
  index($0, "## ") == 1 {
    if (found) exit
    date = substr($0, length(heading) + 1)
    found = index($0, heading) == 1 && date ~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$/
    next
  }
  found { print }
' "${changelog}" | sed '/[^[:space:]]/,$!d')

if [ -z "${notes}" ]; then
  echo "::error::CHANGELOG.md has no non-empty section '## [${version}] - YYYY-MM-DD'" >&2
  exit 1
fi
printf '%s\n' "${notes}"
