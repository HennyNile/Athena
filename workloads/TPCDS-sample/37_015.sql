
select count(*)
 from item, inventory, date_dim, catalog_sales
 where i_current_price between 42 and 42 + 30
 and inv_item_sk = i_item_sk
 and d_date_sk=inv_date_sk
 and d_date between cast('2000-04-02' as date) and (cast('2000-04-02' as date) +  interval '60 day')
 and i_manufact_id in (977,889,772,962)
 and inv_quantity_on_hand between 100 and 500
 and cs_item_sk = i_item_sk;


