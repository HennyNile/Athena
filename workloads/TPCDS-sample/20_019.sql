
select  count(*)
 from	catalog_sales
     ,item 
     ,date_dim
 where cs_item_sk = i_item_sk 
   and i_category in ('Music', 'Books', 'Jewelry')
   and cs_sold_date_sk = d_date_sk
 and d_date between cast('2000-02-25' as date) 
 				and (cast('2000-02-25' as date) + 30 days);


