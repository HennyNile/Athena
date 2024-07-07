
select  count(*)
 from item, inventory, date_dim, store_sales
 where i_current_price between 55 and 55+30
 and inv_item_sk = i_item_sk
 and d_date_sk=inv_date_sk
 and d_date between cast('2001-01-20' as date) and (cast('2001-01-20' as date) +  60 days)
 and i_manufact_id in (217,100,618,147)
 and inv_quantity_on_hand between 100 and 500
 and ss_item_sk = i_item_sk;


