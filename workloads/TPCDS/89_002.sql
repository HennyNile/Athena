
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2000) and
        ((i_category in ('Music','Books','Children') and
          i_class in ('classical','history','school-uniforms')
         )
      or (i_category in ('Women','Electronics','Home') and
          i_class in ('fragrances','scanners','kids') 
        ));


