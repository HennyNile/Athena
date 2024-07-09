
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (1999) and
        ((i_category in ('Books','Children','Home') and
          i_class in ('mystery','toddlers','mattresses')
         )
      or (i_category in ('Electronics','Men','Music') and
          i_class in ('audio','sports-apparel','pop') 
        ));


