from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.core.database import get_async_session
from app.models.database import Person
from faker import Faker
import random

router = APIRouter()

@router.post("/seed-demo")
async def seed_demo_data(
    session: AsyncSession = Depends(get_async_session),
):
    """Seed database with demo data."""
    fake = Faker()
    
    result = await session.execute(text("SELECT COUNT(*) FROM persons"))
    count = result.scalar()
    if count > 100:
        return {"message": f"Database already has {count} records"}
    
    for _ in range(1000):
        person = Person(
            first_name=fake.first_name(),
            last_name=fake.last_name(),
            birth_date=fake.date_of_birth(minimum_age=18, maximum_age=90),
            city=fake.city(),
            state=fake.state_abbr(),
            ssn_last4=str(random.randint(1000, 9999)),
        )
        session.add(person)
    
    await session.commit()
    return {"message": "Seeded 1000 demo records"}
