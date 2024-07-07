
select  count(*)
 from	catalog_sales
     ,item 
     ,date_dim
 where cs_item_sk = i_item_sk 
   and i_category in ('Books', 'Children', 'Music')
   and cs_sold_date_sk = d_date_sk
 and d_date between cast('1999-04-26' as date) 
 				and (cast('1999-04-26' as date) + 30 days);


