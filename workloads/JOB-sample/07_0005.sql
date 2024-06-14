SELECT MIN(an.name) AS acress_pseudonym,
       MIN(t.title) AS japanese_anime_movie
FROM aka_name AS an,
     cast_info AS ci,
     company_name AS cn,
     movie_companies AS mc,
     name AS n,
     role_type AS rt,
     title AS t
WHERE TRUE
  AND cn.country_code = '[us]'
  AND mc.note LIKE '%(TV)%'
  AND mc.note NOT LIKE '%(1994)%'
  AND (mc.note LIKE '%(1994)%'
       OR mc.note LIKE '%(200%)%')
  AND n.name LIKE 'X%'
  AND n.name NOT LIKE '%Yu%'
  AND rt.role = 'actress'
  AND TRUE
  AND TRUE
  AND an.person_id = n.id
  AND n.id = ci.person_id
  AND ci.movie_id = t.id
  AND t.id = mc.movie_id
  AND mc.company_id = cn.id
  AND ci.role_id = rt.id
  AND an.person_id = ci.person_id
  AND ci.movie_id = mc.movie_id;