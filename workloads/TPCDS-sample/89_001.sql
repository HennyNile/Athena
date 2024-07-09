
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2002) and
        ((i_category in ('Sports','Books','Children') and
          i_class in ('fitness','history','newborn')
         )
      or (i_category in ('Music','Women','Home') and
          i_class in ('rock','swimwear','kids') 
        ));


