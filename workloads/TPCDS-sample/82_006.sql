
select  count(*)
 from item, inventory, date_dim, store_sales
 where i_current_price between 4 and 4+30
 and inv_item_sk = i_item_sk
 and d_date_sk=inv_date_sk
 and d_date between cast('2002-03-21' as date) and (cast('2002-03-21' as date) +  interval '60 day')
 and i_manufact_id in (180,989,276,667)
 and inv_quantity_on_hand between 100 and 500
 and ss_item_sk = i_item_sk;


