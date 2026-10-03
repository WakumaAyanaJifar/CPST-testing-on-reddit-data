# Lexicons

Every lexicon used in the paper, extracted directly from the analysis code so the two cannot drift apart. The machine-readable version is `lexicons.json`. Patterns are Python regular expressions, matched case-insensitively.

## Discovery keywords

Used only to find seed authors during collection (Section IV-B). They are not features.

**Core**: `fear of missing out`, `fomo`, `missing out on`, `afraid of missing`, `don't want to miss`

**Comparison**: `everyone else is`, `comparing myself`, `falling behind`, `not keeping up`, `everyone has their life together`, `I'm the only one`, `wasn't invited`, `left out`, `wish I was there`, `could have been`

**Compulsive**: `can't stop checking`, `compulsively checking`, `every few minutes`, `keep opening`, `without thinking`, `lost track of time`, `hours disappeared`, `spent all day scrolling`, `bedtime procrastination`, `wake up and check`

**Platform**: `left on seen`, `story views`, `instagram vs reality`, `just one more video`, `algorithm got me`, `streak anxiety`, `doomscrolling`, `not in the loop`

**Somatic**: `chest tight`, `heart racing`, `can't breathe`, `sick to my stomach`, `couldn't sleep`, `up until 3am`

**Recovery**: `digital detox`, `deleted the app`, `quit instagram`, `taking a break from`, `logging off`, `relapsed`

## Thinking-space lexicons

Each gives a per-record match rate, averaged per author (Section V-D).

| Lexicon | Pattern |
|---|---|
| `self_sg` | `\b(i\|me\|my\|mine\|myself\|i'm\|i've\|i'd\|i'll)\b` |
| `self_pl` | `\b(we\|us\|our\|ours\|ourselves)\b` |
| `other_ref` | `\b(they\|them\|their\|everyone\|everybody\|people\|friends\|others\|someone\|anyone\|others')\b` |
| `second` | `\b(you\|your\|yours\|yourself)\b` |
| `negation` | `\b(not\|no\|never\|none\|nothing\|nobody\|cannot\|can't\|won't\|don't\|doesn't\|didn't\|isn't\|aren't\|wasn't)\b` |
| `absolutist` | `\b(always\|never\|completely\|totally\|entirely\|absolutely\|constantly\|everything\|nothing\|everyone\|nobody\|all the time)\b` |
| `comparative` | `\b(\w+er than\|more than\|less than\|better than\|worse than\|compared to\|comparing\|as much as\|instead of)\b` |
| `past` | `\b(was\|were\|had\|did\|used to\|ago\|yesterday\|last night\|last week\|back then\|should have\|could have\|would have)\b` |
| `future` | `\b(will\|going to\|gonna\|soon\|tomorrow\|next week\|about to\|planning\|upcoming\|later)\b` |
| `uncertainty` | `\b(maybe\|perhaps\|probably\|might\|guess\|sort of\|kind of\|i think\|not sure\|somehow)\b` |
| `exclusion` | `\b(left out\|excluded\|without me\|not invited\|didn't invite\|behind\|missing out\|missed out\|out of the loop)\b` |
| `compulsion` | `\b(keep checking\|can't stop\|refresh\|scrolling\|again and again\|every few minutes\|compulsive\|constantly checking)\b` |
| `somatic` | `\b(heart racing\|chest\|breathe\|breathing\|nausea\|nauseous\|sick to my stomach\|shaking\|dizzy\|exhausted\|headache)\b` |
| `sleep` | `\b(sleep\|asleep\|insomnia\|awake\|3am\|4am\|2am\|all night\|couldn't sleep\|stayed up\|tired)\b` |
| `affect_neg` | `\b(anxious\|anxiety\|sad\|depressed\|lonely\|miserable\|awful\|terrible\|hate\|scared\|afraid\|worried\|upset\|hurt\|guilty\|ashamed\|jealous\|envious\|angry\|frustrated)\b` |
| `affect_pos` | `\b(happy\|glad\|grateful\|proud\|calm\|relieved\|better\|good\|enjoy\|enjoyed\|love\|excited\|hopeful)\b` |

## Convergent-validity probes

Record-level probes for the circadian validation (Section VII-F). Test probes are expected to be more frequent in the estimated local night; control probes are not.

| Probe | Role | Pattern |
|---|---|---|
| sleep terms | test | `\b(sleep\|asleep\|insomnia\|awake\|3am\|4am\|2am\|all night\|couldn't sleep\|stayed up\|tired)\b` |
| sleep, no clock | test | `\b(sleep\|asleep\|insomnia\|awake\|all night\|couldn't sleep\|stayed up\|tired)\b` |
| somatic terms | test | `\b(heart racing\|chest\|breathe\|breathing\|nausea\|nauseous\|sick to my stomach\|shaking\|dizzy\|exhausted\|headache)\b` |
| compulsion terms | test | `\b(keep checking\|can't stop\|refresh\|scrolling\|again and again\|every few minutes\|compulsive\|constantly checking)\b` |
| question marks | control | `\?` |
| second person | control | `\b(you\|your\|yours\|yourself)\b` |
