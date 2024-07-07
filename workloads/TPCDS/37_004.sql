
select count(*)
 from item, inventory, date_dim, catalog_sales
 where i_current_price between 35 and 35 + 30
 and inv_item_sk = i_item_sk
 and d_date_sk=inv_date_sk
 and d_date between cast('2001-07-03' as date) and (cast('2001-07-03' as date) +  60 days)
 and i_manufact_id in (807,880,681,920)
 and inv_quantity_on_hand between 100 and 500
 and cs_item_sk = i_item_sk;


