
select count(*)
 from item, inventory, date_dim, catalog_sales
 where i_current_price between 16 and 16 + 30
 and inv_item_sk = i_item_sk
 and d_date_sk=inv_date_sk
 and d_date between cast('1998-04-25' as date) and (cast('1998-04-25' as date) +  60 days)
 and i_manufact_id in (969,882,799,911)
 and inv_quantity_on_hand between 100 and 500
 and cs_item_sk = i_item_sk;


