filename = 'naturalisasi_januari.csv'
search_keyword = 'pemain diaspora OR pemain naturalisasi OR naturalisasi indonesia -malaysia -vietnam -basket -badminton since:2024-01-01 until:2024-01-31 lang:id'
limit = 1000
!npx -y tweet-harvest@2.6.1 -o "{filename}" -s "{search_keyword}" --tab "LATEST" -l {limit} --token {twitter_auth_token}