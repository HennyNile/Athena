
select  count(*)
 from item, inventory, date_dim, store_sales
 where i_current_price between 77 and 77+30
 and inv_item_sk = i_item_sk
 and d_date_sk=inv_date_sk
 and d_date between cast('2001-07-16' as date) and (cast('2001-07-16' as date) +  60 days)
 and i_manufact_id in (633,939,90,23)
 and inv_quantity_on_hand between 100 and 500
 and ss_item_sk = i_item_sk;


