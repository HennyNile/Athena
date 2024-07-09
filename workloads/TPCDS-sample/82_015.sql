
select  count(*)
 from item, inventory, date_dim, store_sales
 where i_current_price between 65 and 65+30
 and inv_item_sk = i_item_sk
 and d_date_sk=inv_date_sk
 and d_date between cast('2000-05-16' as date) and (cast('2000-05-16' as date) +  interval '60 day')
 and i_manufact_id in (134,345,80,498)
 and inv_quantity_on_hand between 100 and 500
 and ss_item_sk = i_item_sk;


