# mim training section v0.1

Status: draft. This document defines the training section (section kind
5): what a framework such as mimir needs, besides the weights, to resume
training from a file where it stopped. The weights a file holds are the
model at that point; the training section adds the optimizer, its state,
its hyperparameters and the progress counters. The reference
implementation is `training.go`.

The key words MUST, SHOULD and MAY are to be interpreted as described in
RFC 2119.

## 1. Design

- The state is tensors. An optimizer's per-parameter state (Adam's first
  and second moments, a momentum buffer, Adafactor's factored rows and
  columns) and any other state that is an array, such as the random
  generator's, are ordinary tensors in the file's tensor table and weights
  section. They are mapped, aligned and hashed like the weights, and a
  quantized or half-precision moment is just a tensor of that dtype.
- The section is a small directory of names. It says which tensors are
  trainable, which state tensors belong to each, which parameter group
  each is in, and the scalars that are not arrays: hyperparameters per
  group and progress counters.
- Adapters are tensors like any other. When only LoRA adapters are
  trained, only they are listed as parameters; the frozen weights carry no
  state.
- The section does not define the update rule. The optimizer is named, and
  the framework that reads the file implements it. Section 4 lists names
  whose slots and hyperparameters are fixed, so two frameworks that use
  them agree on what the tensors mean.
- Inference never needs it. A writer MUST NOT set the required flag on the
  training section, so a reader that does not implement this document
  still opens the file and runs the model. A tool that strips training
  state removes the section and the tensors it names.

## 2. Layout

All integers are little-endian. All table offsets are relative to the start
of the section and are multiples of 8.

| offset | size | field |
| --- | --- | --- |
| 0 | 4 | magic `MIMT` |
| 4 | 2 | format major (0) |
| 6 | 2 | format minor (1) |
| 8 | 4 | optimizer: string index, 0 when unspecified |
| 12 | 4 | reserved (0) |
| 16 | 112 | directory: 7 entries of (u64 offset, u64 count) |

A reader MUST reject a section whose format major differs from its own. It
MUST accept any minor version of its major; a later minor MAY add tables
by appending directory entries after the seventh, which the reader
ignores.

The directory lists the tables in this fixed order:

| index | table | row size | count means |
| --- | --- | --- | --- |
| 0 | string pool | 1 | bytes |
| 1 | strings | 8 | strings |
| 2 | counters | 16 | counters |
| 3 | hyperparameters | 16 | hyperparameters |
| 4 | parameters | 16 | trainable tensors |
| 5 | states | 8 | state tensors of the parameters |
| 6 | extras | 8 | state tensors not tied to a parameter |

Every table MUST lie inside the section and no count may exceed 2^28.

### 2.1 strings

Row: u32 pool offset, u32 byte length. Each string MUST lie inside the
pool, be valid UTF-8 and be at most 4096 bytes. String 0 MUST be the empty
string. A name below refers to a string by index and MUST NOT refer to an
empty string, except the optimizer, where the empty string means
unspecified. Strings MAY repeat; names are compared by their bytes, not by
their index.

### 2.2 counters

Row: u32 name, u32 reserved (0), u64 value. Names MUST be distinct.

### 2.3 hyperparameters

Row: u32 name, u32 group, f64 value. The pair (name, group) MUST be
distinct. A parameter in group g uses the hyperparameters of group g; a
group is a number, not declared anywhere else, and a group without
hyperparameters uses the optimizer's defaults.

### 2.4 parameters

Row: u32 tensor (name), u32 group, u32 first state, u32 state count. The
tensor is a trainable tensor of the file. A parameter's states are rows
`[first state, first state + state count)` of the states table. The runs
MUST tile the states table in order: the first parameter's run starts at
0, each later run starts where the previous one ended, and the last one
ends at the end of the table.

### 2.5 states

Row: u32 slot (name), u32 tensor (name). Slots MUST be distinct within a
parameter. The tensor holds that slot of the parameter's optimizer state.
Its dtype and shape are the optimizer's business: a moment usually has the
parameter's shape, a factored one does not.

### 2.6 extras

Row: u32 name, u32 tensor (name). Names MUST be distinct. An extra is
state that is not tied to one parameter, such as the random generator's
state as a `u8` tensor, a dynamic loss scale, or a data loader's position.

## 3. Validation

A reader MUST reject the section with `bad-training` if it breaks any rule
of section 2, or if:

1. a tensor named by a parameter, a state or an extra is not in the file's
   tensor table;
2. a tensor is named more than once among the parameters, states and
   extras together: a state is never a parameter, and two slots never
   share a tensor.

Checks against the tensor table need the whole file; the reference reader
makes them in `File.Training` and the others in `DecodeTraining`.

## 4. Well-known names

These names are not required, but a writer that uses one MUST use it with
the meaning given here.

Optimizers, with their slots and hyperparameters:

| optimizer | slots | hyperparameters |
| --- | --- | --- |
| `sgd` | `momentum` (when momentum is not 0) | `lr`, `momentum`, `dampening`, `weight_decay`, `nesterov` (0 or 1) |
| `adam` | `m`, `v`, and `v_max` with AMSGrad | `lr`, `beta1`, `beta2`, `eps`, `weight_decay` (added to the gradient), `amsgrad` (0 or 1) |
| `adamw` | as `adam` | as `adam`, with `weight_decay` decoupled from the gradient |
| `adafactor` | `v_row`, `v_col` for a matrix, `v` otherwise; `m` with momentum | `lr`, `beta1`, `decay_rate`, `eps1`, `eps2`, `clip_threshold` |
| `lion` | `m` | `lr`, `beta1`, `beta2`, `weight_decay` |

`m` and `v` are the exponential moving averages of the gradient and of its
square, with the bias correction applied at use, not stored: a framework
recovers it from the `step` counter.

Counters:

| counter | meaning |
| --- | --- |
| `step` | optimizer steps applied to produce the weights in this file |
| `epoch` | complete passes over the training data |
| `samples` | training examples consumed |
| `tokens` | training tokens consumed |

Extras:

| extra | meaning |
| --- | --- |
| `rng` | the framework's random generator state, as `u8`; its layout is the framework's |
| `loss_scale` | the dynamic loss scale of mixed-precision training, a scalar `f32` |

The reference writer names state tensors `<parameter>@<slot>`, such as
`blk.0.attn_q.weight@m`, and extras `train.<name>`. Other names are
allowed.

## 5. Metadata

Training details that are not state belong in the metadata section under
`train.`: for example `train.dataset` (string), `train.lr_schedule`
(string) or `train.framework` (string, such as `mimir 0.4.1`). They are
informational; the training section is authoritative for the state.
