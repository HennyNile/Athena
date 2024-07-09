
select  count(*)
 from	catalog_sales
     ,item 
     ,date_dim
 where cs_item_sk = i_item_sk 
   and i_category in ('Men', 'Children', 'Books')
   and cs_sold_date_sk = d_date_sk
 and d_date between cast('2001-01-28' as date) 
 				and (cast('2001-01-28' as date) + interval '30 day');


