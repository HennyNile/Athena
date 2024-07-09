
select  count(*)
 from item, inventory, date_dim, store_sales
 where i_current_price between 31 and 31+30
 and inv_item_sk = i_item_sk
 and d_date_sk=inv_date_sk
 and d_date between cast('2000-04-10' as date) and (cast('2000-04-10' as date) +  interval '60 day')
 and i_manufact_id in (543,79,41,7)
 and inv_quantity_on_hand between 100 and 500
 and ss_item_sk = i_item_sk;


