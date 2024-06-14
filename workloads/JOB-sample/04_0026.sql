SELECT MIN(t.title) AS american_vhs_movie
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info AS mi,
     title AS t
WHERE ct.kind = 'distributors'
  AND TRUE
  AND mc.note LIKE '%(theatrical)%'
  AND mc.note LIKE '%(worldwide)%'
  AND TRUE
  AND mi.info = 'Sci-Fi'
  AND t.production_year BETWEEN 1923 AND 1955
  AND t.id = mi.movie_id
  AND t.id = mc.movie_id
  AND mc.movie_id = mi.movie_id
  AND ct.id = mc.company_type_id
  AND it.id = mi.info_type_id;