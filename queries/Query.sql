SELECT id, text, LENGTH(text) as len
FROM embeddings
WHERE text ILIKE '%clapton%'
LIMIT 20;

