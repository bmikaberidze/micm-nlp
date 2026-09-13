# Experiment Orchestration

*group config → many unit runs*

> How do I run many variations, and collect their results together?

```{include} ../../README.md
:start-after: <!-- start:groups -->
:end-before: <!-- end:groups -->
```

## Identity columns

Each run in a group writes the same directory as a {doc}`unit run <config>`.  
Every row of its eval_*.csv and test_*.csv also gets the run's identity: group, name, index, seed, and any extra key you add to the run entry, e.g. `method: spt`.

## Bringing your own science

The default runner is `micm_nlp.pipeline:run`. Replace it with `--runner` when a run needs to do something the pipeline does not.  
E.g. transfer learning: fine-tune the model on one dataset, then test it on another.  
That takes two configs in a single entry: `config` with `mode: finetune`, and `separate_test.config` with `mode: test`.

A runner is a callable with this contract:

```python
def run(config: CONFIG, ctx: RunContext) -> RunOutput:
    ...                                   # tokenizer, dataset, model — whatever your science needs
    trainer = TRAINER(model, dataset, tokenizer)
    return trainer.run()
```

- `config` is the resolved config.  
- `ctx` is a {py:class}`~micm_nlp.group.RunContext`: the group and run names, the index and the entry as written, unknown CLI flags as `extras`, and resolved `test_config` when the entry declared a `separate_test`.  
- The return value is the trainer's {py:class}`~micm_nlp.training.run_output.RunOutput` — the trainer is what writes the results into the run directory.

:::{note}
For the test phase, a second `TRAINER` built from `ctx.test_config` writes its `separate_`-prefixed files into the same directory.
:::

