
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2000) and
        ((i_category in ('Women','Jewelry','Sports') and
          i_class in ('fragrances','earings','hockey')
         )
      or (i_category in ('Books','Shoes','Music') and
          i_class in ('fiction','mens','country') 
        ));


