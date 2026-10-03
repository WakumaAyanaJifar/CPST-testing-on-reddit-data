# Data dictionary

Three files share these columns. `cpst_features.csv` covers each author's full
2023–2025 record set; `cpst_features_h1.csv` and `cpst_features_h2.csv`
recompute every feature on alternating halves of each author's records, for
split-half reliability.

One row is one author-window (2,595 authors with at least 30 records).
Rates marked "per 1k tokens" are matches per 1,000 word tokens across the
author's records. Lexicon patterns are in `lexicons/`.

Features marked **(removed)** are computed but dropped before analysis, because
they are exact functions of other features or, for the offset, a parameter
rather than a measure (paper, Section V-E).

## Identifiers and design variables

| Column | Description |
|---|---|
| `author_hash` | Random author identifier (`a00001`…). Despite the column name, this is **not** a hash: it was assigned at export and cannot be linked to the private data. Consistent across the three files. |
| `cohort` | `seed` (found through a keyword match in an authored record) or `control`. |
| `seed_block` | Community block the author was sampled from: `anxiety` or `disengagement`. |
| `n_items` | In-window records (posts and comments). |
| `tenure_days` | Days between the author's first and last in-window record. |
| `matched` | 1 if the author belongs to a seed–control pair matched on record count and tenure. |
| `pair_id` | Matched-pair identifier, shared by the two members of a pair; −1 if unmatched. Numbering is random. |

## Cyber space (C_)

| Column | Description |
|---|---|
| `C_n_communities` | Distinct communities posted in. |
| `C_community_entropy` | Normalized entropy of the author's records across communities. |
| `C_top_community_share` | Share of records in the author's most-used community. |
| `C_anxiety_share` | Share of records in anxiety-block communities. |
| `C_diseng_share` | Share of records in disengagement-block communities. |
| `C_offblock_share` | Share of records in other communities. **(removed)** |
| `C_post_share` | Share of records that are submissions rather than comments. |
| `C_median_score` | Median platform score of the author's records. |
| `C_neg_score_share` | Share of records with score below 1. |
| `C_selfpost_share` | Share of records that are text (self) posts. |
| `C_n_link_domains` | Distinct outbound link domains. |
| `C_social_link_share` | Share of outbound links pointing to other social platforms. |
| `C_edited_share` | Share of records edited after posting. |
| `C_community_churn` | One minus the Jaccard overlap of communities used in the first and second halves of the window. |
| `C_communities_per_month` | Distinct communities per 30 days of tenure. |

## Physical space (P_)

Local time is inferred per author from the six-hour activity trough, anchored
at 04:00 local (paper, Section V-B).

| Column | Description |
|---|---|
| `P_utc_offset_est` | Estimated UTC offset in hours. **(removed; a parameter, not a feature)** |
| `P_rhythm_trough_depth` | Trough activity relative to mean activity; near 1 means no detectable rhythm. Offsets above 0.75 are treated as untrusted. |
| `P_rhythm_strength` | Circular resultant length of local posting hours. |
| `P_hour_entropy` | Normalized entropy of local posting hours. |
| `P_night_share` | Share of records posted 00:00–05:00 local. |
| `P_latenight_share` | Share of records posted 01:00–04:00 local. |
| `P_workhours_share` | Share of records posted 09:00–18:00 local. |
| `P_weekend_share` | Share of records posted on Saturday or Sunday. |
| `P_median_gap_s` | Median seconds between consecutive records. |
| `P_burst_share` | Share of gaps shorter than five minutes. |
| `P_n_sessions` | Sessions, splitting at gaps longer than 30 minutes. |
| `P_items_per_session` | Records per session. |
| `P_active_days` | Distinct days with at least one record. |
| `P_activity_density` | Active days divided by tenure. |
| `P_items_per_active_day` | Records per active day. |
| `P_reply_latency_med` | Median seconds between another person's record and this author's reply. Defined only where the parent record was collected. |
| `P_fast_reply_share` | Share of such replies made within five minutes. Same coverage as above. |

## Social space (S_)

Partner-based features count only partners whose records are in the analysed
corpus.

| Column | Description |
|---|---|
| `S_comment_share` | Share of records that are comments. **(removed; complement of `C_post_share`)** |
| `S_reply_to_comment_share` | Share of comments replying to a comment rather than to a submission. |
| `S_mean_depth` | Mean thread depth of the author's comments. Defined where depth is known. |
| `S_max_depth` | Maximum thread depth. **(removed; near-duplicate of mean depth)** |
| `S_n_partners` | Distinct authors replied to. |
| `S_partners_per_item` | Distinct partners per record. |
| `S_replies_received` | Replies received from analysed authors. |
| `S_reply_ratio` | Replies received per record. |
| `S_mutual_partners` | Partners with replies in both directions. |
| `S_reciprocity` | Mutual partners divided by all partners. |
| `S_n_threads` | Distinct threads participated in. |
| `S_items_per_thread` | Records per thread. |

## Thinking space (T_)

| Column | Description |
|---|---|
| `T_tokens_total` | Total word tokens. |
| `T_mean_tokens` | Mean tokens per record. |
| `T_median_tokens` | Median tokens per record. |
| `T_len_variability` | Coefficient of variation of record length. |
| `T_type_token_ratio` | Distinct words divided by total tokens. |
| `T_self_sg` | First-person singular pronouns, per 1k tokens. |
| `T_self_pl` | First-person plural pronouns, per 1k tokens. |
| `T_other_ref` | References to others (they, people, friends…), per 1k tokens. |
| `T_second_person` | Second-person pronouns, per 1k tokens. |
| `T_self_other_ratio` | First-person singular count over other-reference count. **(removed)** |
| `T_negation` | Negations, per 1k tokens. |
| `T_absolutist` | Absolutist terms, per 1k tokens. |
| `T_comparative` | Comparative constructions, per 1k tokens. |
| `T_past` | Past-orientation terms, per 1k tokens. |
| `T_future` | Future-orientation terms, per 1k tokens. |
| `T_temporal_balance` | (future − past) / (future + past). **(removed)** |
| `T_uncertainty` | Uncertainty markers, per 1k tokens. |
| `T_exclusion_lex` | Exclusion vocabulary, per 1k tokens. |
| `T_compulsion_lex` | Compulsive-checking vocabulary, per 1k tokens. |
| `T_somatic_lex` | Somatic vocabulary, per 1k tokens. |
| `T_sleep_lex` | Sleep vocabulary, per 1k tokens. |
| `T_affect_neg` | Negative affect terms, per 1k tokens. |
| `T_affect_pos` | Positive affect terms, per 1k tokens. |
| `T_affect_balance` | (positive − negative) / (positive + negative). **(removed)** |
| `T_question_rate` | Question marks, per 1k tokens. |
| `T_exclaim_rate` | Exclamation marks, per 1k tokens. |
| `T_ellipsis_rate` | Ellipses, per 1k tokens. |
