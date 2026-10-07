"""Run the unchanged arm algorithm with atomic caches and completed-fit recovery."""
from pathlib import Path

from homeroom.arms import ArmContext
from homeroom.checkpoint import atomic_json, checkpointed_train
from homeroom.config import generated_dir
from homeroom.run import main
from homeroom.teacher import TeacherSession


def run():
    original = ArmContext.train

    def train(ctx, texts, labels):
        return checkpointed_train(ctx.cfg, ctx.seed, ctx.task.n_classes, texts, labels,
                                  lambda: original(ctx, texts, labels),
                                  generated_dir(ctx.cfg) / "student_checkpoints" / f"seed{ctx.seed}")

    def store(session, key, task, reply):
        atomic_json(session.cache_dir / f"{key}.json",
                    {"task": task, "text": reply.text, "in": reply.usage.input_tokens,
                     "out": reply.usage.output_tokens})

    ArmContext.train = train
    TeacherSession._store = store
    main()


if __name__ == "__main__":
    run()
