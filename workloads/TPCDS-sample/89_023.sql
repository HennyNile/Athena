
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (1999) and
        ((i_category in ('Women','Children','Music') and
          i_class in ('maternity','school-uniforms','classical')
         )
      or (i_category in ('Men','Books','Sports') and
          i_class in ('shirts','history','fitness') 
        ));


