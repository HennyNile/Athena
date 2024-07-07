
select  count(*)
  from inventory
      ,warehouse
      ,item
      ,date_dim
  where i_current_price between 0.99 and 1.49
    and i_item_sk          = inv_item_sk
    and inv_warehouse_sk   = w_warehouse_sk
    and inv_date_sk    = d_date_sk
    and d_date between (cast ('2002-03-30' as date) - 30 days)
                  and (cast ('2002-03-30' as date) + 30 days);


