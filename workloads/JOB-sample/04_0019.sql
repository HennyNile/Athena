SELECT MIN(t.title) AS american_vhs_movie
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info AS mi,
     title AS t
WHERE ct.kind = 'distributors'
  
  AND mc.note LIKE '%(2007)%'
  
  
  AND mi.info IN ('Action', 'American', 'Bulgaria')
  AND t.production_year BETWEEN 1957 AND 1974
  AND t.id = mi.movie_id
  AND t.id = mc.movie_id
  AND mc.movie_id = mi.movie_id
  AND ct.id = mc.company_type_id
  AND it.id = mi.info_type_id;