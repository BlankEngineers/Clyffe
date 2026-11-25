# Check 6 — Tokenizer Round-Trip Audit (`clyffe_v0.1_tokenizer`)

| Input | Encoded IDs | Tokens | Decoded |
|---|---|---|---|
| `hello` | `[64, 331]` | `['h', 'ello']` | `'hello'` |
| `hi` | `[64, 65]` | `['h', 'i']` | `'hi'` |
| `Hello!` | `[481, 26]` | `['Hello', '!']` | `'Hello!'` |
| `Good morning` | `[180, 358]` | `['Good', '▁morning']` | `'Good morning'` |
| `who are you` | `[79, 64, 71, 191, 101]` | `['w', 'h', 'o', '▁are', '▁you']` | `'who are you'` |
| `what is your name` | `[79, 64, 98, 104, 348, 244]` | `['w', 'h', 'at', '▁is', '▁your', '▁name']` | `'what is your name'` |
| `My name is Hari.` | `[426, 244, 104, 479, 33]` | `['My', '▁name', '▁is', '▁Hari', '.']` | `'My name is Hari.'` |

**Findings:** Round-trip preserves all characters (`hello` -> `hello`, no off-by-one deletion). However, case-sensitivity splits lowercase `hello` into `['h', 'ello']` while capitalized `Hello` is a single token `[481]`.

