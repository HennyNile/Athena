
select  count(*)
 from date_dim, store_sales, item
 where d_date_sk = ss_sold_date_sk
 	and ss_item_sk = i_item_sk
 	and i_manager_id=31
 	and d_moy=12
 	and d_year=1998;


