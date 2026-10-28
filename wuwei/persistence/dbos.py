"""Optional DBOS scheduling adapter; Wuwei's store remains execution truth.

Register the workflow before DBOS.launch(). DBOS persists deliveries/steps;
Wuwei still refuses replay of an unacknowledged executing job after a crash.
"""

from __future__ import annotations

from .jobs import JobService


class DBOSJobBackend:
    """Use an injected DBOSClient and queue without taking over its lifecycle."""

    def __init__(
        self, service: JobService, client, *, queue_name="wuwei", workflow_name="wuwei_job"
    ):
        self.service, self.client = service, client
        self.queue_name, self.workflow_name = queue_name, workflow_name

    async def _enqueue(self, job):
        # Include revision: each approval continuation is a distinct durable delivery.
        await self.client.enqueue_async(
            {
                "queue_name": self.queue_name,
                "workflow_name": self.workflow_name,
                "workflow_id": f"{self.service.store.namespace}:{job.id}:{job.revision}",
            },
            job.id,
        )
        return job

    async def submit(self, user_input, *, idempotency_key=None, session_id=None):
        job = await self.service.store.submit(
            user_input, idempotency_key=idempotency_key, session_id=session_id
        )
        if job.status == "queued":
            await self._enqueue(job)
        return job

    async def get(self, job_id):
        return await self.service.store.get(job_id)

    async def cancel(self, job_id):
        # Do not kill the DBOS workflow first: our worker must finish recording
        # cancellation/unknown truth. Its heartbeat observes this durable request.
        return await self.service.store.cancel(job_id)

    async def signal(self, job_id, *, approved, operator, signal_id, reason=""):
        job = await self.service.store.signal(
            job_id, approved=approved, operator=operator, signal_id=signal_id, reason=reason
        )
        if job.status == "queued":
            await self._enqueue(job)
        return job


def build_dbos_workflow(service: JobService, *, name="wuwei_job", dbos=None):
    """Register a workflow/step using the official DBOS decorators.

    Application owns DBOS config, queue, database, launch and destroy. Agent
    calls run inside a step; nondeterministic APIs never run in workflow code.
    """
    if dbos is None:
        try:
            from dbos import DBOS as dbos
        except ImportError as exc:
            raise ImportError("Install wuwei[dbos] to register a DBOS worker") from exc

    @dbos.step(name=f"{name}_execute")
    async def execute(job_id):
        record = await service.execute(job_id)
        return record.model_dump(mode="json")

    @dbos.workflow(name=name)
    async def workflow(job_id):
        return await execute(job_id)

    return workflow
