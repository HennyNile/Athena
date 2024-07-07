
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2000) and
        ((i_category in ('Home','Shoes','Music') and
          i_class in ('blinds/shades','womens','pop')
         )
      or (i_category in ('Books','Children','Electronics') and
          i_class in ('entertainments','newborn','memory') 
        ));


