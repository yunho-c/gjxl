#!/usr/bin/env python3
"""Resume the recorded October 7 timing study with its documented retry amendment.

The original controller stays immutable. Only the environment-rejected attempt
budget changes; all acceptance criteria, samples, ordering and timing gates stay
identical. Requires an explicitly amended, provenance-checked manifest.
"""
import timing

timing.PROTOCOL['maximum_attempts_per_pair'] = 12

if __name__ == '__main__':
    timing.main()
