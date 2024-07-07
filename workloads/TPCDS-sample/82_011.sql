
select  count(*)
 from item, inventory, date_dim, store_sales
 where i_current_price between 81 and 81+30
 and inv_item_sk = i_item_sk
 and d_date_sk=inv_date_sk
 and d_date between cast('1998-06-13' as date) and (cast('1998-06-13' as date) +  60 days)
 and i_manufact_id in (751,190,160,636)
 and inv_quantity_on_hand between 100 and 500
 and ss_item_sk = i_item_sk;


