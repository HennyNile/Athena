SELECT MIN(t.title) AS american_vhs_movie
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info AS mi,
     title AS t
WHERE ct.kind = 'distributors'
  
  AND mc.note LIKE '%(VHS)%'
  
  
  AND mi.info IN ('Action', 'American', 'Bulgaria', 'Crime', 'Denish', 'Denmark', 'Drama', 'English', 'Family')
  AND t.production_year >= 1997
  AND t.id = mi.movie_id
  AND t.id = mc.movie_id
  AND mc.movie_id = mi.movie_id
  AND ct.id = mc.company_type_id
  AND it.id = mi.info_type_id;