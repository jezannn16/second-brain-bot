# Demo: Second Brain

```sh
python second_brain.py --db /tmp/second-brain-demo.sqlite3 note 'Командный SQL' 'Для аналитики данных применяйте SQL; проверяйте результат на тестовых данных.' --tags work,analytics
python second_brain.py --db /tmp/second-brain-demo.sqlite3 search 'аналитики SQL' --tag work
python second_brain.py --db /tmp/second-brain-demo.sqlite3 ask 'SQL аналитика'
python second_brain.py --db /tmp/second-brain-demo.sqlite3 export
```

Expected behavior: search returns a matching snippet with its note id and source; `ask` returns that passage as a citation. A question with no matching terms returns an explicit no-source response. Delete the record with `delete ID`; subsequent search should no longer show it.
