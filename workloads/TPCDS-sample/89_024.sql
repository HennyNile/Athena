
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2002) and
        ((i_category in ('Children','Sports','Shoes') and
          i_class in ('toddlers','basketball','womens')
         )
      or (i_category in ('Books','Electronics','Men') and
          i_class in ('business','karoke','pants') 
        ));


