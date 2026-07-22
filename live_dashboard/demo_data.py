"""Synthetic Turkish events and tenant-specific coaching configuration."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.coaching.rule_engine import CoachingRule
from app.events.models import (
    CoachingAction,
    SuggestionPriority,
    TranscriptEvent,
    TranscriptKind,
)
from app.tenancy.models import (
    TenantASRConfig,
    TenantClassificationConfig,
    TenantCoachingConfig,
    TenantConfig,
    TenantContext,
    TenantRAGConfig,
)


DEMO_START = datetime(2026, 7, 22, 9, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class DemoScenario:
    scenario_id: str
    name: str
    events: tuple[TranscriptEvent, ...]


@dataclass(frozen=True, slots=True)
class TenantDemo:
    config: TenantConfig
    rules: tuple[CoachingRule, ...]
    scenarios: tuple[DemoScenario, ...]


def _event(
    tenant_id: str,
    call_id: str,
    revision: int,
    kind: TranscriptKind,
    text: str,
    start: float,
    end: float,
) -> TranscriptEvent:
    return TranscriptEvent(
        tenant_id=tenant_id,
        call_id=call_id,
        event_id=f"{tenant_id}-demo-{revision}",
        kind=kind,
        text=text,
        start_seconds=start,
        end_seconds=end,
        revision=revision,
        created_at_utc=DEMO_START + timedelta(seconds=end),
    )


def _scenario(
    tenant_id: str,
    scenario_id: str,
    name: str,
    phrases: tuple[tuple[TranscriptKind, str], ...],
) -> DemoScenario:
    call_id = "demo-call"
    events = tuple(
        _event(tenant_id, call_id, index, kind, text, (index - 1) * 4.0, index * 4.0)
        for index, (kind, text) in enumerate(phrases, start=1)
    )
    return DemoScenario(scenario_id, name, events)


def _config(tenant_id: str, tenant_name: str, labels: list[str]) -> TenantConfig:
    return TenantConfig(
        context=TenantContext(tenant_id=tenant_id, tenant_name=tenant_name),
        asr=TenantASRConfig(),
        classification=TenantClassificationConfig(
            model_id=f"{tenant_id}-demo-rules", labels=labels
        ),
        rag=TenantRAGConfig(enabled=False),
        coaching=TenantCoachingConfig(
            cooldown_seconds=8.0,
            max_active_suggestions=2,
            enable_llm=False,
            allowed_actions=[action.value for action in CoachingAction],
        ),
    )


def _rule(
    rule_id: str,
    label: str,
    phrase: str,
    title: str,
    suggestion: str,
    *,
    priority: SuggestionPriority = SuggestionPriority.MEDIUM,
    action: CoachingAction = CoachingAction.TEMPLATE_ACTION,
    exclude: tuple[str, ...] = (),
) -> CoachingRule:
    return CoachingRule(
        rule_id=rule_id,
        label=label,
        include_any=(phrase,),
        exclude_any=exclude,
        action=action,
        priority=priority,
        title=title,
        suggestion=suggestion,
        evidence_ids=(f"synthetic-{rule_id}",),
    )


def tenant_demos() -> dict[str, TenantDemo]:
    """Return fresh, isolated synthetic data for each tenant."""
    alpha = "tenant_alpha"
    beta = "tenant_beta"
    common_scenarios_alpha = (
        _scenario(
            alpha,
            "product",
            "Ürün bilgisi",
            (
                (TranscriptKind.PARTIAL, "Paket özelliklerini"),
                (TranscriptKind.STABLE, "Paket özelliklerini öğrenmek istiyorum."),
            ),
        ),
        _scenario(
            alpha,
            "price",
            "Fiyat itirazı",
            (
                (TranscriptKind.PARTIAL, "Bu ücret bana"),
                (TranscriptKind.STABLE, "Bu ücret bana çok pahalı geldi."),
                (TranscriptKind.FINAL, "Bu ücret bana çok pahalı geldi."),
            ),
        ),
        _scenario(
            alpha,
            "cancel",
            "İptal talebi",
            (
                (TranscriptKind.PARTIAL, "Aboneliğimi"),
                (TranscriptKind.STABLE, "Aboneliğimi iptal etmek istiyorum."),
                (TranscriptKind.STABLE, "Aboneliğimi iptal etmek istiyorum."),
            ),
        ),
        _scenario(
            alpha,
            "negated",
            "Olumsuzlanmış iptal",
            (
                (TranscriptKind.PARTIAL, "İptal etmek"),
                (
                    TranscriptKind.FINAL,
                    "İptal etmek istemiyorum, yalnızca bilgi alıyorum.",
                ),
            ),
        ),
        _scenario(
            alpha,
            "critical",
            "Kritik eskalasyon",
            (
                (TranscriptKind.PARTIAL, "Acil olarak"),
                (TranscriptKind.STABLE, "Acil olarak yetkiliyle görüşmek istiyorum."),
            ),
        ),
    )
    alpha_rules = (
        _rule(
            "a-product",
            "urun_bilgisi",
            "özelliklerini öğrenmek",
            "Ürün bilgisini açıklayın",
            "Paket özelliklerini kısa ve anlaşılır biçimde özetleyin.",
        ),
        _rule(
            "a-price",
            "fiyat_itirazi",
            "çok pahalı",
            "Fiyat itirazını karşılayın",
            "İhtiyacı netleştirip uygun seçenekleri açıklayın.",
            priority=SuggestionPriority.HIGH,
        ),
        _rule(
            "a-cancel",
            "iptal_riski",
            "iptal etmek istiyorum",
            "İptal talebini doğrulayın",
            "Talebi teyit edin ve onaylı iptal adımlarını açıklayın.",
            priority=SuggestionPriority.HIGH,
            exclude=("iptal etmek istemiyorum",),
        ),
        _rule(
            "a-critical",
            "kritik_eskalasyon",
            "acil olarak yetkiliyle",
            "Acil eskalasyon",
            "Görüşmeyi gecikmeden yetkili ekibe aktarın.",
            priority=SuggestionPriority.CRITICAL,
            action=CoachingAction.ESCALATE,
        ),
    )
    beta_scenarios = tuple(
        DemoScenario(
            item.scenario_id,
            item.name,
            tuple(
                event.model_copy(
                    update={
                        "tenant_id": beta,
                        "event_id": event.event_id.replace(alpha, beta),
                    }
                )
                for event in item.events
            ),
        )
        for item in common_scenarios_alpha
    )
    beta_rules = (
        _rule(
            "b-product",
            "paket_sorusu",
            "özelliklerini öğrenmek",
            "Paket karşılaştırması",
            "Sentetik paket seçeneklerini karşılaştırın.",
        ),
        _rule(
            "b-price",
            "butce_endisesi",
            "çok pahalı",
            "Bütçe endişesi",
            "Bütçe aralığını sorun ve sentetik alternatifleri sunun.",
            priority=SuggestionPriority.HIGH,
        ),
        _rule(
            "b-cancel",
            "ayrilma_talebi",
            "iptal etmek istiyorum",
            "Ayrılma talebi",
            "Talebi doğrulayın ve tarafsız süreç bilgisini verin.",
            priority=SuggestionPriority.HIGH,
            exclude=("iptal etmek istemiyorum",),
        ),
        _rule(
            "b-critical",
            "yonetici_aktarimi",
            "acil olarak yetkiliyle",
            "Yönetici aktarımı",
            "Temsilci güvenlik adımlarını izleyerek yöneticiye aktarsın.",
            priority=SuggestionPriority.CRITICAL,
            action=CoachingAction.ESCALATE,
        ),
    )
    return {
        alpha: TenantDemo(
            _config(
                alpha,
                "Sentetik Alfa",
                ["urun_bilgisi", "fiyat_itirazi", "iptal_riski", "kritik_eskalasyon"],
            ),
            alpha_rules,
            common_scenarios_alpha,
        ),
        beta: TenantDemo(
            _config(
                beta,
                "Sentetik Beta",
                [
                    "paket_sorusu",
                    "butce_endisesi",
                    "ayrilma_talebi",
                    "yonetici_aktarimi",
                ],
            ),
            beta_rules,
            beta_scenarios,
        ),
    }


def scenario_for(tenant_id: str, scenario_id: str) -> DemoScenario:
    demo = tenant_demos()[tenant_id]
    return next(item for item in demo.scenarios if item.scenario_id == scenario_id)
