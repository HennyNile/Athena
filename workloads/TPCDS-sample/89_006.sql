
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2001) and
        ((i_category in ('Men','Music','Jewelry') and
          i_class in ('sports-apparel','country','bracelets')
         )
      or (i_category in ('Children','Shoes','Home') and
          i_class in ('newborn','athletic','tables') 
        ));


