SELECT id, text, LENGTH(text) as len
FROM embeddings
WHERE text ILIKE '%IFBB%'
LIMIT 20;

