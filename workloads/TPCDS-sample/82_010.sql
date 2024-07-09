
select  count(*)
 from item, inventory, date_dim, store_sales
 where i_current_price between 17 and 17+30
 and inv_item_sk = i_item_sk
 and d_date_sk=inv_date_sk
 and d_date between cast('1999-02-13' as date) and (cast('1999-02-13' as date) +  interval '60 day')
 and i_manufact_id in (994,547,751,560)
 and inv_quantity_on_hand between 100 and 500
 and ss_item_sk = i_item_sk;


